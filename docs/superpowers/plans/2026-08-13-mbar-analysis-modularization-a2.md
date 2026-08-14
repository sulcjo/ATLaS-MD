# MBAR Analysis Modularization — Plan A2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Relocate the `Data` dataclass and every data-loading function in `analyze_gareus_mbar.py` (~35 functions) into four new modules under `gareus/mbar_analysis/`, as Plan A2 of the 6-plan modularization sequence. Pure relocation — no behavior, signature, or name changes.

**Architecture:** `gareus/mbar_analysis/data.py` (Data dataclass, JSON/CSV primitives, temp/beta inference, Data-transform and epoch/regime helpers) sits at the bottom of an acyclic dependency chain. `gareus/mbar_analysis/loaders_adaptive.py` (adaptive-production directory discovery, vectorized index remapping, legacy union-NPZ/epoch-CSV loaders, rounds augmentation) depends only on `data.py`. `gareus/mbar_analysis/loaders_union_parquet.py` (the primary adaptive-production union-Parquet loader) depends on `data.py` and `loaders_adaptive.py`. `gareus/mbar_analysis/loaders.py` (NPZ/CSV/Parquet single-run loaders, `prod_dir_of`, the `load_data` dispatcher) sits at the top, depending on all three. Four bias-reconstruction functions (`_compute_u_nk_analytical`, `_parse_epoch_window_map_native_params`, `_epoch_bias_param_vectors`, `_reconstruct_union_bias_block`) stay behind in `analyze_gareus_mbar.py` for Plan A3; the two moved loaders that need them use a lazy, function-body-local reverse import (matching this file's own existing `from gareus.query import load_samples` idiom) to avoid a circular import. `analyze_gareus_mbar.py` re-exports every relocated name via four top-of-file import blocks, mirroring Plan A1's `from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K` pattern.

**Tech Stack:** Python 3.10+, pytest, numpy. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a2-design.md`

## Global Constraints

- **Plan A1 must already be executed** before this plan starts: `gareus/mbar_analysis/__init__.py` must exist, and `gareus/units.py` must already export `KJ_PER_KCAL`/`K_B_KJ_PER_MOL_K`. Task 1 Step 0 verifies this.
- This is a pure relocation: every moved function/class keeps its exact current name, signature, and body. No behavior change of any kind.
- No static linter (`pyflakes`/`ruff`) is available in this environment (verified: `ModuleNotFoundError: No module named 'pyflakes'`; `which ruff` empty). `python -m py_compile` only catches syntax errors, not a `NameError` buried in a function body that's never called (e.g. forgetting to move a private helper class a function's `return` statement depends on). Every module-creation task therefore requires a test that actually **invokes** each newly-added function/class against a minimal real fixture — not just imports it — before that task is considered done.
- The reverse-direction imports of `_compute_u_nk_analytical`/`_parse_epoch_window_map_native_params`/`_epoch_bias_param_vectors`/`_reconstruct_union_bias_block` from `analyze_gareus_mbar` **must** be function-body-local (lazy), never at module top — a module-top reverse import creates a real circular `ImportError` (traced in the spec's Architecture section), not just a style issue.
- `analyze_gareus_mbar.py`'s old invocation (`python analyze_gareus_mbar.py <run_dir> ...`) must keep working unchanged throughout — every rewire task re-exports every relocated name so the remaining ~9,900 lines need zero changes beyond the import block itself.
- Object-identity (`is`) checks, not re-derived behavioral tests, are the primary proof that a rewire task actually replaced a local definition with an import (see spec Testing #1 for why: `clean`/`_apply_analysis_stride`/`_skip_first_n_frames`/`_filter_epoch_source` all mutate their `Data` argument in place, so a naive call-twice-and-compare test can pass even when comparing a mutated object against itself).

---

### Task 1: Create `gareus/mbar_analysis/data.py` — Part A: JSON/CSV primitives + `Data` dataclass

**Files:**
- Create: `gareus/mbar_analysis/data.py`
- Test: `tests/test_mbar_analysis_data_part_a.py` (new file)

**Interfaces:**
- Consumes: `gareus.units.KJ_PER_KCAL`, `gareus.units.K_B_KJ_PER_MOL_K` (Plan A1).
- Produces: `gareus.mbar_analysis.data.rjson`, `.wjson`, `._json_default`, `.read_windows`, `.jvec`, `.Data` — consumed by later tasks in this plan and by every downstream loader module.

- [ ] **Step 0: Verify Plan A1 has been executed**

Run: `python3 -c "import gareus.mbar_analysis; from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K; print('A1 OK')"`
Expected: prints `A1 OK`. If this fails, stop — Plan A1 must be implemented first (this plan is not implementable standalone).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_data_part_a.py`:

```python
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.data import rjson, wjson, _json_default, read_windows, jvec, Data


def test_rjson_missing_file_returns_default():
    assert rjson(Path("/nonexistent/path/does/not/exist.json")) == {}
    assert rjson(Path("/nonexistent/path/does/not/exist.json"), default=[]) == []


def test_rjson_reads_real_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}')
    assert rjson(p) == {"a": 1}


def test_wjson_writes_and_rjson_reads_back(tmp_path):
    p = tmp_path / "sub" / "x.json"
    wjson(p, {"a": np.float64(1.5), "b": np.int64(3)})
    assert rjson(p) == {"a": 1.5, "b": 3}


def test_json_default_handles_numpy_types():
    assert _json_default(np.array([1, 2])) == [1, 2]
    assert _json_default(np.int64(5)) == 5
    assert _json_default(np.float64(float("nan"))) is None
    assert _json_default(np.bool_(True)) is True
    assert _json_default(Path("/a/b")) == "/a/b"


def test_read_windows_parses_center_and_k(tmp_path):
    p = tmp_path / "umbrella_windows.csv"
    with p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["center_A", "k_kcal_mol_A2"])
        w.writeheader()
        w.writerow({"center_A": "1.0", "k_kcal_mol_A2": "10.0"})
        w.writerow({"center_A": "2.0", "k_kcal_mol_A2": "10.0"})
    centers, ks, rows = read_windows(p)
    assert list(centers) == [1.0, 2.0]
    assert list(ks) == [10.0, 10.0]
    assert len(rows) == 2


def test_read_windows_missing_file_returns_empty():
    centers, ks, rows = read_windows(Path("/nonexistent/umbrella_windows.csv"))
    assert centers.size == 0 and ks.size == 0 and rows == []


def test_jvec_parses_json_list_and_dict():
    assert jvec("[1.0, 2.0]") == [1.0, 2.0]
    assert jvec(json.dumps({"1": 2.0, "0": 1.0})) == [1.0, 2.0]
    assert jvec(None) == []
    assert jvec("") == []


def test_data_dataclass_holds_all_fields():
    d = Data(
        prod_dir=Path("/p"), out_dir=Path("/o"),
        cv=np.array([1.0]), cv2=np.array([np.nan]), rg_A=np.array([np.nan]),
        window=np.array([0]), replica=np.array([0]), step=np.array([0]),
        u_nk=np.array([[0.0]]), centers=np.array([1.0]), k_kcal=np.array([10.0]),
        beta=1.0, temp=300.0, boost_kj=np.array([np.nan]), potential_kj=None,
        source="test", meta={},
    )
    assert d.boost_dih_kj is None
    assert d.beta == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_data_part_a.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.mbar_analysis.data'`

- [ ] **Step 3: Write minimal implementation**

Create `gareus/mbar_analysis/data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_data_part_a.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/data.py tests/test_mbar_analysis_data_part_a.py
git commit -m "feat: add gareus.mbar_analysis.data Part A (JSON/CSV primitives + Data dataclass)"
```

---

### Task 2: Extend `gareus/mbar_analysis/data.py` — Part B: temp/beta inference + `Data`-transform helpers

**Files:**
- Modify: `gareus/mbar_analysis/data.py` (append)
- Test: `tests/test_mbar_analysis_data_part_b.py` (new file)

**Interfaces:**
- Consumes: `Data` (Task 1).
- Produces: `gareus.mbar_analysis.data.infer_temp_beta`, `.clean`, `._masked_data`, `.
  _apply_analysis_stride`, `._filter_epoch_source`, `._sample_block_ids`, `._skip_first_n_frames`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_data_part_b.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from gareus.mbar_analysis.data import (
    Data, K_B_KJ_PER_MOL_K, infer_temp_beta, clean, _masked_data,
    _apply_analysis_stride, _filter_epoch_source, _sample_block_ids,
    _skip_first_n_frames,
)


def _mk_data(n=6, k=2, meta=None):
    rng = np.arange(n, dtype=float)
    return Data(
        prod_dir=Path("/p"), out_dir=Path("/o"),
        cv=rng, cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=np.zeros(n, int), replica=np.array([i % 2 for i in range(n)]),
        step=np.arange(n), u_nk=np.zeros((n, k)),
        centers=np.zeros(k), k_kcal=np.zeros(k),
        beta=1.0, temp=300.0, boost_kj=np.full(n, np.nan), potential_kj=None,
        source="test", meta=meta if meta is not None else {},
    )


def test_infer_temp_beta_reads_meta_top_level(tmp_path):
    meta = {"temperature_K": 310.0}
    temp, beta = infer_temp_beta(tmp_path, meta)
    assert temp == 310.0
    assert beta == pytest.approx(1.0 / (K_B_KJ_PER_MOL_K * 310.0))


def test_infer_temp_beta_reads_nested_resolved_args(tmp_path):
    meta = {"resolved_args": {"temperature_k": 305.0}}
    temp, beta = infer_temp_beta(tmp_path, meta)
    assert temp == 305.0


def test_infer_temp_beta_falls_back_to_300k_with_warning(tmp_path):
    with pytest.warns(RuntimeWarning):
        temp, beta = infer_temp_beta(tmp_path, {})
    assert temp == 300.0


def test_clean_drops_nonfinite_cv_rows():
    d = _mk_data(n=4)
    d.cv[1] = float("nan")
    out = clean(d)
    assert out.cv.size == 3
    assert not np.any(np.isnan(out.cv))


def test_clean_raises_on_shape_mismatch():
    d = _mk_data(n=4, k=2)
    d.u_nk = np.zeros((3, 2))
    with pytest.raises(ValueError):
        clean(d)


def test_masked_data_slices_per_sample_arrays_shares_window_arrays():
    d = _mk_data(n=6)
    mask = np.array([True, False, True, False, True, False])
    out = _masked_data(d, mask)
    assert out.cv.size == 3
    assert out.centers is d.centers  # window-space arrays shared, not sliced
    assert out.beta == d.beta


def test_apply_analysis_stride_keeps_every_nth_sample_per_replica():
    d = _mk_data(n=8)
    d.replica = np.zeros(8, int)
    d.step = np.arange(8)
    out = _apply_analysis_stride(d, stride=2, offset=0)
    assert out.cv.size == 4
    assert list(out.step) == [0, 2, 4, 6]


def test_apply_analysis_stride_noop_for_stride_one():
    d = _mk_data(n=5)
    out = _apply_analysis_stride(d, stride=1, offset=0)
    assert out is d


def test_filter_epoch_source_filters_meta_list_in_place():
    d = _mk_data(n=4, meta={"_epoch_source": [0, 0, 1, 1]})
    keep = np.array([True, False, True, True])
    _filter_epoch_source(d, keep)
    assert d.meta["_epoch_source"] == [0, 1, 1]


def test_sample_block_ids_groups_by_epoch_source_and_replica():
    d = _mk_data(n=4, meta={"_epoch_source": [0, 0, 1, 1]})
    d.replica = np.array([0, 0, 0, 0])
    block_ids = _sample_block_ids(d)
    # Same replica, different epoch source -> different blocks.
    assert block_ids[0] == block_ids[1]
    assert block_ids[0] != block_ids[2]
    assert block_ids[2] == block_ids[3]


def test_sample_block_ids_falls_back_to_replica_when_epoch_source_absent():
    d = _mk_data(n=4)
    d.replica = np.array([0, 1, 0, 1])
    block_ids = _sample_block_ids(d)
    assert block_ids[0] == block_ids[2]
    assert block_ids[1] == block_ids[3]
    assert block_ids[0] != block_ids[1]


def test_skip_first_n_frames_drops_earliest_steps_per_replica():
    d = _mk_data(n=6)
    d.replica = np.array([0, 0, 0, 1, 1, 1])
    d.step = np.array([0, 1, 2, 0, 1, 2])
    out = _skip_first_n_frames(d, 1)
    assert out.cv.size == 4
    assert sorted(out.step[out.replica == 0]) == [1, 2]
    assert sorted(out.step[out.replica == 1]) == [1, 2]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_data_part_b.py -v`
Expected: FAIL with `ImportError: cannot import name 'infer_temp_beta' from 'gareus.mbar_analysis.data'`

- [ ] **Step 3: Write minimal implementation**

Append to `gareus/mbar_analysis/data.py` (after the `Data` dataclass; note the test above imports `K_B_KJ_PER_MOL_K` directly from this module, which already works since Part A's module-top `from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K` makes it a module attribute):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_data_part_b.py -v`
Expected: PASS (12 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/data.py tests/test_mbar_analysis_data_part_b.py
git commit -m "feat: add gareus.mbar_analysis.data Part B (temp/beta inference + Data-transform helpers)"
```

---

### Task 3: Extend `gareus/mbar_analysis/data.py` — Part C: epoch/regime metadata helpers

**Files:**
- Modify: `gareus/mbar_analysis/data.py` (append)
- Test: `tests/test_mbar_analysis_data_part_c.py` (new file)

**Interfaces:**
- Consumes: `Data` (Task 1).
- Produces: `gareus.mbar_analysis.data._epoch_dir_index`, `._epoch_number_for_run_dir`, `._epoch_run_manifest_secondary_cv_type`, `._epoch_zero_split_masks`, `._secondary_cv_epoch_regime_masks` — `_epoch_dir_index` is consumed by `loaders_adaptive.py` (Task 5).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_data_part_c.py`:

```python
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.data import (
    Data, _epoch_dir_index, _epoch_number_for_run_dir,
    _epoch_run_manifest_secondary_cv_type, _epoch_zero_split_masks,
    _secondary_cv_epoch_regime_masks,
)


def _mk_data(n, epoch_src, run_dirs, cv=None):
    return Data(
        prod_dir=Path("/p"), out_dir=Path("/o"),
        cv=cv if cv is not None else np.arange(n, dtype=float),
        cv2=np.full(n, np.nan), rg_A=np.full(n, np.nan),
        window=np.zeros(n, int), replica=np.zeros(n, int), step=np.arange(n),
        u_nk=np.zeros((n, 1)), centers=np.zeros(1), k_kcal=np.zeros(1),
        beta=1.0, temp=300.0, boost_kj=np.full(n, np.nan), potential_kj=None,
        source="test",
        meta={"_epoch_source": epoch_src, "adaptive_epoch_run_dirs": [str(r) for r in run_dirs]},
    )


def test_epoch_dir_index_matches_epoch_nnn():
    assert _epoch_dir_index(Path("/a/epoch_003")) == 3
    assert _epoch_dir_index(Path("/a/final")) is None


def test_epoch_number_for_run_dir_checks_flat_and_subrun_layout():
    assert _epoch_number_for_run_dir(Path("/a/epoch_002")) == 2
    assert _epoch_number_for_run_dir(Path("/a/epoch_002/baseline")) == 2
    assert _epoch_number_for_run_dir(Path("/a/final/baseline")) is None


def test_epoch_run_manifest_secondary_cv_type_reads_resolved_args(tmp_path):
    (tmp_path / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}), encoding="utf-8")
    assert _epoch_run_manifest_secondary_cv_type(tmp_path) == "torsion-pca"


def test_epoch_run_manifest_secondary_cv_type_missing_file_returns_empty(tmp_path):
    assert _epoch_run_manifest_secondary_cv_type(tmp_path / "nope") == ""


def test_epoch_zero_split_masks_splits_epoch0_from_rest(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    d = _mk_data(4, [0, 0, 1, 1], [e0, e1])
    result = _epoch_zero_split_masks(d)
    assert result is not None
    mask0, mask_rest = result
    assert list(mask0) == [True, True, False, False]
    assert list(mask_rest) == [False, False, True, True]


def test_epoch_zero_split_masks_returns_none_without_epoch_source():
    d = _mk_data(4, None, [])
    d.meta = {}
    assert _epoch_zero_split_masks(d) is None


def test_secondary_cv_epoch_regime_masks_detects_two_regimes(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}), encoding="utf-8")
    e1 = tmp_path / "epoch_001"; e1.mkdir()
    (e1 / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "tica-linear"}}), encoding="utf-8")
    d = _mk_data(4, [0, 0, 1, 1], [e0, e1])
    regimes = _secondary_cv_epoch_regime_masks(d)
    assert regimes is not None
    assert set(regimes.keys()) == {"torsion-pca", "tica-linear"}
    # tica-linear is chronologically last -> dominant.
    assert regimes["tica-linear"][1] is True
    assert regimes["torsion-pca"][1] is False


def test_secondary_cv_epoch_regime_masks_returns_none_for_single_regime(tmp_path):
    e0 = tmp_path / "epoch_000"; e0.mkdir()
    (e0 / "run_manifest.json").write_text(
        json.dumps({"resolved_args": {"secondary_cv": "torsion-pca"}}), encoding="utf-8")
    d = _mk_data(2, [0, 0], [e0])
    assert _secondary_cv_epoch_regime_masks(d) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_data_part_c.py -v`
Expected: FAIL with `ImportError: cannot import name '_epoch_dir_index' from 'gareus.mbar_analysis.data'`

- [ ] **Step 3: Write minimal implementation**

Append to `gareus/mbar_analysis/data.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_data_part_c.py -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/data.py tests/test_mbar_analysis_data_part_c.py
git commit -m "feat: add gareus.mbar_analysis.data Part C (epoch/regime metadata helpers)"
```

---

### Task 4: Rewire `analyze_gareus_mbar.py` to import from `gareus.mbar_analysis.data`

**Files:**
- Modify: `analyze_gareus_mbar.py` (delete the 17 relocated definitions listed below; add one import block)
- Test: `tests/test_analyze_gareus_mbar_uses_mbar_analysis_data.py` (new file)

**Interfaces:**
- Consumes: everything produced by Tasks 1-3.
- Produces: `analyze_gareus_mbar.Data`, `.rjson`, `.wjson`, `.read_windows`, `.jvec`, `.infer_temp_beta`, `.clean`, `._masked_data`, `._apply_analysis_stride`, `._filter_epoch_source`, `._sample_block_ids`, `._skip_first_n_frames`, `._epoch_dir_index`, `._epoch_number_for_run_dir`, `._epoch_run_manifest_secondary_cv_type`, `._epoch_zero_split_masks`, `._secondary_cv_epoch_regime_masks` still exist as module attributes (unchanged names, now imported) — consumed unchanged by every one of the ~9,900 remaining lines in this file, and by Tasks 5-9 (which rely on `analyze_gareus_mbar` still exposing these under their original names, since the deferred bias-math functions call back into this same namespace).

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_gareus_mbar_uses_mbar_analysis_data.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.mbar_analysis.data as mdata
import analyze_gareus_mbar as agm

_NAMES = [
    "Data", "rjson", "wjson", "read_windows", "jvec", "infer_temp_beta",
    "clean", "_masked_data", "_apply_analysis_stride", "_filter_epoch_source",
    "_sample_block_ids", "_skip_first_n_frames", "_epoch_dir_index",
    "_epoch_number_for_run_dir", "_epoch_run_manifest_secondary_cv_type",
    "_epoch_zero_split_masks", "_secondary_cv_epoch_regime_masks",
]


def test_every_relocated_data_name_is_the_shared_object():
    for name in _NAMES:
        assert getattr(agm, name) is getattr(mdata, name), (
            f"{name}: analyze_gareus_mbar still defines its own copy "
            f"instead of importing gareus.mbar_analysis.data's"
        )
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyze_gareus_mbar_uses_mbar_analysis_data.py -v`
Expected: FAIL — every `assert ... is ...` fails (`analyze_gareus_mbar.py` still has its own local `def`/`class` for each name, so `agm.Data is mdata.Data` etc. are all `False`).

- [ ] **Step 3: Write minimal implementation**

In `analyze_gareus_mbar.py`:

1. Re-locate the current definitions to confirm exact ranges (line numbers below are approximate as of this session and WILL have drifted; re-run this before editing):

```bash
grep -n "^class Data:\|^def rjson\|^def wjson\|^def _json_default\|^def read_windows\|^def jvec\|^def infer_temp_beta\|^def clean(\|^def _masked_data\|^def _apply_analysis_stride\|^def _filter_epoch_source\|^def _sample_block_ids\|^def _skip_first_n_frames\|^def _epoch_dir_index\|^def _epoch_number_for_run_dir\|^def _epoch_run_manifest_secondary_cv_type\|^def _epoch_zero_split_masks\|^def _secondary_cv_epoch_regime_masks" analyze_gareus_mbar.py
```

2. Immediately after the `from gareus.units import KJ_PER_KCAL, K_B_KJ_PER_MOL_K` line Plan A1 added, insert:

```python
from gareus.mbar_analysis.data import (
    Data, rjson, wjson, read_windows, jvec, infer_temp_beta,
    clean, _masked_data, _apply_analysis_stride, _filter_epoch_source,
    _sample_block_ids, _skip_first_n_frames, _epoch_dir_index,
    _epoch_number_for_run_dir, _epoch_run_manifest_secondary_cv_type,
    _epoch_zero_split_masks, _secondary_cv_epoch_regime_masks,
)
```

   (`_json_default` is intentionally not re-exported: `grep -n "_json_default" analyze_gareus_mbar.py` confirms it is referenced nowhere in this file outside its own now-deleted definition — it was only ever called by `wjson`, which lives in `gareus.mbar_analysis.data` now too.)

3. Delete each of the 17 original definitions found in Step 1 (the full `class Data:` block including its docstring-free field list, and each `def <name>(...):` function body up to its next top-level `def`/`class`/blank-separator). Do not reorder or edit anything else in the file — every other function's body is untouched, since they reference these 17 names by bare identifier and Python resolves that against whatever the name is bound to at call time (the new import), regardless of textual position in the file.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyze_gareus_mbar_uses_mbar_analysis_data.py -v`
Expected: PASS (1 test, 17 assertions)

- [ ] **Step 5: Confirm the rest of the file still works**

Run: `python -m py_compile analyze_gareus_mbar.py`
Expected: clean compile.

Run: `pytest -q tests/test_infer_temp_beta.py tests/test_pmf_bootstrap_uncertainty.py tests/test_epoch0_pmf_gamd_split.py tests/test_masked_logw_subset_pmf.py tests/test_secondary_cv_regime_split.py -v`
Expected: PASS, same as before this task (these are the existing test files that exercise the 17 relocated names through `analyze_gareus_mbar`'s namespace; per the Global Constraints, they need zero code changes).

Run: `pytest -q tests/ --ignore=tests/test_validation_common.py -k "not physics_oracle and not thermodynamic_validity"`
Expected: same pre-existing failures as before this task, no new ones.

- [ ] **Step 6: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_analyze_gareus_mbar_uses_mbar_analysis_data.py
git commit -m "refactor: import Data-loading-domain names from gareus.mbar_analysis.data instead of redefining"
```

---

### Task 5: Create `gareus/mbar_analysis/loaders_adaptive.py` — Part A: directory discovery + vectorized lookup

**Files:**
- Create: `gareus/mbar_analysis/loaders_adaptive.py`
- Test: `tests/test_mbar_analysis_loaders_adaptive_part_a.py` (new file)

**Interfaces:**
- Consumes: `gareus.mbar_analysis.data._epoch_dir_index` (Task 3).
- Produces: `gareus.mbar_analysis.loaders_adaptive._find_adaptive_epoch_dirs`, `._find_selfcontained_epoch_dirs`, `._find_adaptive_epoch_csv_sources`, `._has_epoch_csv_layout`, `._find_adaptive_final_run_dirs`, `._find_gareus_round_dirs`, `._vectorized_map_lookup`, `._vectorized_map_lookup_or_self`, `._vectorized_map_index` — consumed by `loaders.py` (Task 8) and `loaders_union_parquet.py` (Task 7).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_loaders_adaptive_part_a.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.loaders_adaptive import (
    _find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs,
    _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout,
    _find_adaptive_final_run_dirs, _find_gareus_round_dirs,
    _vectorized_map_lookup, _vectorized_map_lookup_or_self, _vectorized_map_index,
)


def test_find_adaptive_epoch_dirs_finds_flat_layout(tmp_path):
    ep = tmp_path / "epoch_000"
    (ep / "samples").mkdir(parents=True)
    (ep / "segments.json").write_text("[]")
    (ep / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")
    result = _find_adaptive_epoch_dirs(tmp_path)
    assert len(result) == 1
    assert result[0][0] == ep


def test_find_adaptive_epoch_dirs_finds_scheduled_baseline_topup_layout(tmp_path):
    ep = tmp_path / "epoch_001"
    ep.mkdir()
    (ep / "epoch_window_map.csv").write_text("epoch_window,state_id\n0,0\n")
    base = ep / "baseline"
    (base / "samples").mkdir(parents=True)
    (base / "segments.json").write_text("[]")
    result = _find_adaptive_epoch_dirs(tmp_path)
    assert len(result) == 1
    assert result[0][0] == base


def test_find_selfcontained_epoch_dirs_requires_windows_dir(tmp_path):
    ep = tmp_path / "epoch_000"
    (ep / "samples").mkdir(parents=True)
    (ep / "segments.json").write_text("[]")
    (ep / "windows").mkdir()
    assert _find_selfcontained_epoch_dirs(tmp_path) == [ep]


def test_find_adaptive_epoch_csv_sources_and_has_epoch_csv_layout(tmp_path):
    ep = tmp_path / "epoch_000"
    ep.mkdir()
    (ep / "samples.csv").write_text("cv_A\n1.0\n")
    assert _has_epoch_csv_layout(tmp_path) is True
    assert _find_adaptive_epoch_csv_sources(tmp_path) == [ep]


def test_find_adaptive_final_run_dirs_finds_final_and_topup(tmp_path):
    final = tmp_path / "final"
    (final / "baseline").mkdir(parents=True)
    (final / "baseline" / "samples.csv").write_text("cv_A\n1.0\n")
    (final / "topup_001_1000").mkdir()
    (final / "topup_001_1000" / "samples.csv").write_text("cv_A\n1.0\n")
    result = _find_adaptive_final_run_dirs(tmp_path)
    assert final / "baseline" in result
    assert final / "topup_001_1000" in result


def test_find_gareus_round_dirs_requires_windows_csv_and_chunks(tmp_path):
    rd = tmp_path / "adaptive_feedback_round_01"
    (rd / "analysis_chunks").mkdir(parents=True)
    (rd / "analysis_chunks" / "chunk_000.npz").write_bytes(b"")
    (rd / "umbrella_windows.csv").write_text("center_A,k_kcal_mol_A2\n1.0,10.0\n")
    assert _find_gareus_round_dirs(tmp_path) == [rd]


def test_vectorized_map_lookup_matches_naive_dict_get():
    arr = np.array([0, 1, 2, 5])
    mapping = {0: 10, 1: 11, 2: 12}
    out = _vectorized_map_lookup(arr, mapping, default=-1, dtype=np.int64)
    assert list(out) == [10, 11, 12, -1]


def test_vectorized_map_lookup_or_self_falls_back_to_identity():
    arr = np.array([0, 1, 7])
    mapping = {0: 100}
    out = _vectorized_map_lookup_or_self(arr, mapping, dtype=np.int64)
    assert list(out) == [100, 1, 7]


def test_vectorized_map_index_raises_keyerror_on_missing_key():
    import pytest
    arr = np.array([0, 1, 9])
    mapping = {0: 10, 1: 11}
    with pytest.raises(KeyError):
        _vectorized_map_index(arr, mapping, dtype=np.int64)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_loaders_adaptive_part_a.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.mbar_analysis.loaders_adaptive'`

- [ ] **Step 3: Write minimal implementation**

Create `gareus/mbar_analysis/loaders_adaptive.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_loaders_adaptive_part_a.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/loaders_adaptive.py tests/test_mbar_analysis_loaders_adaptive_part_a.py
git commit -m "feat: add gareus.mbar_analysis.loaders_adaptive Part A (directory discovery + vectorized lookup)"
```

---

### Task 6: Extend `gareus/mbar_analysis/loaders_adaptive.py` — Part B: legacy loaders + rounds augmentation

**Files:**
- Modify: `gareus/mbar_analysis/loaders_adaptive.py` (append)
- Test: `tests/test_mbar_analysis_loaders_adaptive_part_b.py` (new file)

**Interfaces:**
- Consumes: `Data`, `clean`, `infer_temp_beta`, `rjson`, `read_windows` (Task 1-2); `_find_adaptive_epoch_csv_sources`, `_find_adaptive_final_run_dirs`, `_find_gareus_round_dirs`, `_vectorized_map_lookup`, `_vectorized_map_lookup_or_self` (Task 5); `analyze_gareus_mbar._compute_u_nk_analytical` (stays in the script, Plan A3's — consumed via a **lazy, function-body-local** import inside `_augment_with_adaptive_rounds` only).
- Produces: `gareus.mbar_analysis.loaders_adaptive.load_epoch_csv_adaptive`, `.load_union_npz`, `._load_round_raw`, `._build_union_window_table`, `._round_window_to_union_map`, `._augment_with_adaptive_rounds` — the last three (plus `load_union_npz`, `load_epoch_csv_adaptive`) consumed by `loaders.py`'s `load_data` dispatcher (Task 8).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_loaders_adaptive_part_b.py`:

```python
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from gareus.mbar_analysis.loaders_adaptive import (
    load_epoch_csv_adaptive, load_union_npz, _load_round_raw,
    _build_union_window_table, _round_window_to_union_map,
    _augment_with_adaptive_rounds,
)


def _write_samples_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_load_epoch_csv_adaptive_unions_windows_across_epoch_sources(tmp_path):
    ap = tmp_path / "adaptive_production"
    fields = ["cv_A", "primary_cv_center", "primary_cv_k", "step", "replica"]
    _write_samples_csv(
        ap / "epoch_000" / "samples.csv",
        [{"cv_A": "1.0", "primary_cv_center": "1.0", "primary_cv_k": "10.0", "step": "0", "replica": "0"}],
        fields,
    )
    _write_samples_csv(
        ap / "epoch_001" / "samples.csv",
        [{"cv_A": "2.0", "primary_cv_center": "2.0", "primary_cv_k": "10.0", "step": "0", "replica": "0"}],
        fields,
    )
    (ap.parent / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_epoch_csv_adaptive(ap)
    assert d.cv.size == 2
    assert d.u_nk.shape == (2, 2)
    assert d.meta["source"] == "epoch_csv_adaptive"


def test_load_union_npz_reads_arrays_and_infers_beta(tmp_path):
    ap = tmp_path / "adaptive_production"
    ap.mkdir(parents=True)
    np.savez(
        ap / "adaptive_union_mbar.npz",
        cv_A=np.array([1.0, 2.0]),
        secondary_cv=np.array([np.nan, np.nan]),
        sampled_state_ids=np.array([0, 1]),
        umbrella_reduced_bias_nk=np.zeros((2, 2)),
        primary_centers=np.array([1.0, 2.0]),
        primary_k=np.array([10.0, 10.0]),
    )
    (ap.parent / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_union_npz(ap)
    assert d.cv.size == 2
    assert d.meta["source"] == "adaptive_union_mbar"


def test_load_round_raw_concatenates_chunks(tmp_path):
    rd = tmp_path / "adaptive_feedback_round_01"
    chunks = rd / "analysis_chunks"
    chunks.mkdir(parents=True)
    np.savez(chunks / "chunk_000.npz", cv_A=np.array([1.0, 2.0]))
    np.savez(chunks / "chunk_001.npz", cv_A=np.array([3.0]))
    raw = _load_round_raw(rd)
    assert raw is not None
    assert list(raw["cv_A"]) == [1.0, 2.0, 3.0]
    assert raw["window"].size == 3


def test_build_union_window_table_deduplicates_by_tolerance(tmp_path):
    d1 = tmp_path / "d1"; d1.mkdir()
    (d1 / "umbrella_windows.csv").write_text(
        "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,1.0,10.0,1.0,10.0\n")
    d2 = tmp_path / "d2"; d2.mkdir()
    (d2 / "umbrella_windows.csv").write_text(
        "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,1.00001,10.0,1.00001,10.0\n"
        "1,2.0,10.0,2.0,10.0\n")
    table = _build_union_window_table([d1, d2])
    assert len(table) == 2  # d1's row and d2's first row merge (within tol)


def test_round_window_to_union_map_maps_local_to_union_index(tmp_path):
    rd = tmp_path / "round"; rd.mkdir()
    (rd / "umbrella_windows.csv").write_text(
        "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,2.0,10.0,2.0,10.0\n")
    union_windows = [
        {"primary_center": 1.0, "secondary_cv_center": float("nan"), "primary_k_kcal": 10.0, "secondary_k_kcal": 0.0},
        {"primary_center": 2.0, "secondary_cv_center": float("nan"), "primary_k_kcal": 10.0, "secondary_k_kcal": 0.0},
    ]
    mapping = _round_window_to_union_map(rd, union_windows)
    assert mapping == {0: 1}


def test_augment_with_adaptive_rounds_returns_original_when_no_round_dirs(tmp_path, monkeypatch):
    import gareus.mbar_analysis.data as mdata
    d = mdata.Data(
        prod_dir=tmp_path, out_dir=tmp_path, cv=np.array([1.0]), cv2=np.array([np.nan]),
        rg_A=np.array([np.nan]), window=np.array([0]), replica=np.array([0]),
        step=np.array([0]), u_nk=np.zeros((1, 1)), centers=np.array([1.0]),
        k_kcal=np.array([10.0]), beta=1.0, temp=300.0, boost_kj=np.array([np.nan]),
        potential_kj=None, source="test", meta={},
    )
    out = _augment_with_adaptive_rounds(d, tmp_path)
    assert out is d


def test_augment_with_adaptive_rounds_merges_round_samples_with_final(tmp_path):
    """Real invocation of the full merge path (not just the no-round-dirs
    early return above) -- exercises the lazy
    `from analyze_gareus_mbar import _compute_u_nk_analytical` import inside
    the function body. At this point in the plan sequence,
    analyze_gareus_mbar.py still defines _compute_u_nk_analytical itself
    (Plan A3 relocates it later), so this import resolves against the
    original, untouched definition."""
    import gareus.mbar_analysis.data as mdata

    run_dir = tmp_path
    windows_csv = "window,center_A,k_kcal_mol_A2,primary_center,primary_k\n0,1.0,10.0,1.0,10.0\n"
    (run_dir / "umbrella_windows.csv").write_text(windows_csv)
    round_dir = run_dir / "adaptive_feedback_round_01"
    round_dir.mkdir()
    (round_dir / "umbrella_windows.csv").write_text(windows_csv)
    chunks = round_dir / "analysis_chunks"
    chunks.mkdir()
    np.savez(
        chunks / "chunk_000.npz",
        cv_A=np.array([1.1, 1.2]), step=np.array([0, 1]), replica=np.array([0, 0]),
        window=np.array([0, 0]), gamd_boost_total_kj_mol=np.array([0.5, 0.6]),
        potential_kj_mol=np.array([-10.0, -11.0]),
    )
    d = mdata.Data(
        prod_dir=run_dir, out_dir=run_dir, cv=np.array([1.0]), cv2=np.array([np.nan]),
        rg_A=np.array([np.nan]), window=np.array([0]), replica=np.array([0]),
        step=np.array([0]), u_nk=np.zeros((1, 1)), centers=np.array([1.0]),
        k_kcal=np.array([10.0]), beta=1.0, temp=300.0, boost_kj=np.array([np.nan]),
        potential_kj=None, source="test", meta={},
    )

    out = _augment_with_adaptive_rounds(d, run_dir)

    assert out.cv.size == 3  # 1 final-production sample + 2 round samples
    assert out.u_nk.shape == (3, 1)
    assert np.all(np.isfinite(out.u_nk))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_loaders_adaptive_part_b.py -v`
Expected: FAIL with `ImportError: cannot import name 'load_epoch_csv_adaptive' from 'gareus.mbar_analysis.loaders_adaptive'`

- [ ] **Step 3: Write minimal implementation**

Append to `gareus/mbar_analysis/loaders_adaptive.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_loaders_adaptive_part_b.py -v`
Expected: PASS (7 tests). `test_augment_with_adaptive_rounds_merges_round_samples_with_final` is the one that actually exercises the lazy `from analyze_gareus_mbar import _compute_u_nk_analytical` import inside the function body — at this point in the plan, `analyze_gareus_mbar.py` still defines that function itself (Plan A3 relocates it later), so the import resolves correctly against the original definition. This is the real proof the lazy-import design in the spec's Architecture section actually works, not just that it compiles.

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/loaders_adaptive.py tests/test_mbar_analysis_loaders_adaptive_part_b.py
git commit -m "feat: add gareus.mbar_analysis.loaders_adaptive Part B (legacy loaders + rounds augmentation)"
```

---

### Task 7: Create `gareus/mbar_analysis/loaders_union_parquet.py`

**Files:**
- Create: `gareus/mbar_analysis/loaders_union_parquet.py`
- Test: `tests/test_mbar_analysis_loaders_union_parquet.py` (new file)

**Interfaces:**
- Consumes: `Data`, `clean`, `infer_temp_beta`, `rjson` (Task 1-2); `_find_adaptive_epoch_dirs`, `_vectorized_map_lookup`, `_vectorized_map_index` (Task 5); `gareus.query.load_samples` (pre-existing, unchanged, imported lazily inside functions exactly as it is today); `analyze_gareus_mbar._parse_epoch_window_map_native_params`/`._epoch_bias_param_vectors`/`._reconstruct_union_bias_block` (stay in the script, Plan A3's — consumed via **lazy, function-body-local** imports).
- Produces: `gareus.mbar_analysis.loaders_union_parquet.load_parquet_adaptive_union` — consumed by `loaders.py`'s `load_data` dispatcher (Task 8). `_is_usable_for_mbar`/`_merge_missing_usable_states` are internal to this module only (already confirmed via `grep` that no other function outside this cluster calls them).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_loaders_union_parquet.py`:

```python
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from gareus.mbar_analysis.loaders_union_parquet import (
    _is_usable_for_mbar, _merge_missing_usable_states, load_parquet_adaptive_union,
)


def test_is_usable_for_mbar_accepts_true_variants():
    assert _is_usable_for_mbar({"usable_for_mbar": "True"}) is True
    assert _is_usable_for_mbar({"usable_for_mbar": "1"}) is True
    assert _is_usable_for_mbar({"usable_for_mbar": "yes"}) is True
    assert _is_usable_for_mbar({"usable_for_mbar": "false"}) is False
    assert _is_usable_for_mbar({}) is False


def test_merge_missing_usable_states_adds_only_usable_missing_rows():
    primary = [{"state_id": "0"}]
    live = [
        {"state_id": "0", "usable_for_mbar": "True"},
        {"state_id": "1", "usable_for_mbar": "True"},
        {"state_id": "2", "usable_for_mbar": "False"},
    ]
    merged = _merge_missing_usable_states(primary, live)
    ids = {int(r["state_id"]) for r in merged}
    assert ids == {0, 1}


def _write_registry(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["state_id", "usable_for_mbar", "primary_center", "primary_k",
              "secondary_center", "secondary_k", "burnin_steps"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _write_epoch_window_map(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["epoch_window", "state_id", "primary_center", "primary_k",
              "secondary_center", "secondary_k"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_load_parquet_adaptive_union_raises_without_registry(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        load_parquet_adaptive_union(tmp_path)


def test_load_parquet_adaptive_union_raises_without_epoch_dirs(tmp_path):
    import pytest
    _write_registry(tmp_path / "final_registry_used_for_mbar.csv", [
        {"state_id": "0", "usable_for_mbar": "True", "primary_center": "1.0",
         "primary_k": "10.0", "secondary_center": "", "secondary_k": "", "burnin_steps": "0"},
    ])
    with pytest.raises(FileNotFoundError):
        load_parquet_adaptive_union(tmp_path)


def _write_epoch_with_real_parquet(epoch_dir, samples, state_id=0, primary_center=0.0,
                                    primary_k=44.3, secondary_center=0.0, secondary_k=0.0):
    """Write one epoch's worth of real Parquet samples + window snapshot +
    epoch_window_map.csv, using the real gareus.store writers -- same fixture
    pattern already used by tests/test_perf_loading_and_solver_memory.py's
    _write_epoch and tests/test_union_mbar_per_epoch_bias.py's _write_epoch."""
    from gareus.store import ParquetSampleWriter, SegmentRegistry, WindowSnapshot

    epoch_dir.mkdir(parents=True, exist_ok=True)
    reg = SegmentRegistry(epoch_dir)
    seg_id = reg.open_segment("run_001", None, 1)
    WindowSnapshot(epoch_dir).snapshot(
        seg_id, [{"window_id": 0, "center1": primary_center, "k1": primary_k,
                  "center2": secondary_center, "k2": secondary_k}],
        cv1_type="contacts", cv2_type="torsion-pca",
    )
    writer = ParquetSampleWriter(epoch_dir / "samples" / seg_id, flush_rows=1000)
    max_step = 0
    for step, replica, cv1, cv2 in samples:
        writer.write_sample(step, replica, 0, cv1, cv2, -100.0, 5.0, 2.0, 0.4)
        max_step = max(max_step, step)
    writer.close()
    reg.close_segment(seg_id, end_step=max_step)

    (epoch_dir / "epoch_window_map.csv").write_text(
        "epoch_window,state_id,primary_center,primary_k,secondary_center,secondary_k\n"
        f"0,{state_id},{primary_center},{primary_k},{secondary_center},{secondary_k}\n",
        encoding="utf-8",
    )


def test_load_parquet_adaptive_union_end_to_end_with_real_parquet_fixture(tmp_path):
    """Real invocation of the full success path -- exercises _load_epoch_task
    and both lazy reverse-imports of the four bias-reconstruction functions
    that stay behind in analyze_gareus_mbar.py for Plan A3. This is the test
    that would have caught a forgotten import (e.g. of _Arrays in loaders.py,
    or of the bias-math cluster here) at module-creation time rather than
    only surfacing later in Task 9's full-suite run."""
    adaptive_dir = tmp_path / "adaptive_production"
    adaptive_dir.mkdir(parents=True)
    samples = [(i * 50, i % 2, float(i) * 0.1, 0.0) for i in range(10)]
    _write_epoch_with_real_parquet(adaptive_dir / "epoch_000", samples, state_id=0, primary_center=0.0)
    _write_registry(adaptive_dir / "final_registry_used_for_mbar.csv", [
        {"state_id": "0", "usable_for_mbar": "True", "primary_center": "0.0",
         "primary_k": "44.3", "secondary_center": "", "secondary_k": "", "burnin_steps": "0"},
    ])
    (adaptive_dir.parent / "run_manifest.json").write_text(
        '{"resolved_args": {"temperature_k": 300.0}}', encoding="utf-8")

    d = load_parquet_adaptive_union(adaptive_dir)

    assert d.cv.size == 10
    assert d.u_nk.shape == (10, 1)
    assert np.all(np.isfinite(d.u_nk))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_loaders_union_parquet.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.mbar_analysis.loaders_union_parquet'`

- [ ] **Step 3: Write minimal implementation**

Create `gareus/mbar_analysis/loaders_union_parquet.py`:

```python
"""Data-loading domain: the primary adaptive-production union-Parquet loader.

Relocated verbatim from ``analyze_gareus_mbar.py`` (Plan A2). Kept in its own
module, separate from ``loaders_adaptive.py``'s legacy/fallback adaptive
loaders, because this is the loader this repo's CLAUDE.md documents an
entire historical bug narrative around (the per-epoch-native-bias fix for
mid-campaign window recentering) -- see
docs/superpowers/specs/2026-08-13-mbar-analysis-modularization-a2-design.md.
"""
from __future__ import annotations

import csv
import math
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.units import K_B_KJ_PER_MOL_K
from .data import Data, clean, infer_temp_beta, rjson
from .loaders_adaptive import _find_adaptive_epoch_dirs, _vectorized_map_lookup, _vectorized_map_index


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


def _load_epoch_task(epoch_dir: Path, wmap_path: Path, n_threads: int) -> tuple:
    """Load one epoch's samples, window map, and native bias params (runs in a thread)."""
    from gareus.query import load_samples
    # NOTE (Plan A2 -> Plan A3 handoff): _parse_epoch_window_map_native_params
    # is one of four bias-reconstruction functions deliberately left behind in
    # analyze_gareus_mbar.py for Plan A3's audited reconciliation with
    # gareus.query.reconstruct_bias_matrix. This import MUST stay
    # function-body-local (lazy) for the same circular-import reason
    # documented in loaders_adaptive.py's _augment_with_adaptive_rounds; when
    # Plan A3 relocates this function, update the module path in this line.
    from analyze_gareus_mbar import _parse_epoch_window_map_native_params
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
    # NOTE (Plan A2 -> Plan A3 handoff): both of these stay behind in
    # analyze_gareus_mbar.py for Plan A3; see _load_epoch_task's own note
    # above for why this import must be lazy.
    from analyze_gareus_mbar import _epoch_bias_param_vectors, _reconstruct_union_bias_block

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
        # Vectorized equivalent of [wmap.get(int(w), -1) for w in raw_w] -- see
        # _vectorized_map_lookup docstring; bit-identical output including the
        # -1 sentinel for keys absent from wmap.
        remapped = _vectorized_map_lookup(raw_w, wmap, default=-1, dtype=np.int32)
        valid = remapped >= 0
        if not np.any(valid):
            continue
        # Filter first, cast second: avoids allocating a full-epoch-length
        # float64 transient that's then mostly discarded by [valid].
        cv_epoch = samples['cv1'][valid].astype(np.float64, copy=False)
        all_cv.append(cv_epoch)
        cv2_raw = samples.get('cv2')
        cv2_epoch = cv2_raw[valid].astype(np.float64, copy=False) if cv2_raw is not None else np.full(valid.sum(), np.nan)
        all_cv2.append(cv2_epoch)
        # Vectorized equivalent of [state_id_to_k[int(s)] for s in remapped[valid]]
        # -- see _vectorized_map_index docstring; raises KeyError on a missing
        # state_id exactly like the original dict subscripting did. Narrowed to
        # int16 (Data.window's on-disk source is uint16; downstream consumers
        # already defensively re-cast to int64 before use -- see CLAUDE.md/audit).
        all_window.append(_vectorized_map_index(remapped[valid], state_id_to_k, dtype=np.int16))
        all_step.append(samples['step'][valid].astype(np.int64, copy=False))
        # NOTE: intentionally NOT applying the filter-then-cast reorder here --
        # the `else` branch already builds an array sized to valid.sum() (not
        # the full epoch length), and rebasing it on `[valid]` after slicing
        # would require restructuring around a latent shape mismatch in that
        # branch (np.zeros(int(valid.sum()), ...) then indexed again by the
        # full-length `valid` mask) that is out of scope to touch here.
        rep = samples['replica'].astype(np.int16) if 'replica' in samples else np.zeros(int(valid.sum()), np.int16)
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

    # Free each per-epoch block list right after it's concatenated -- these
    # hold the same data twice (once per-epoch, once pooled) until GC'd, and
    # this is the dominant contributor to the loader's peak memory footprint
    # for large multi-epoch runs.
    cv      = np.concatenate(all_cv);      del all_cv
    cv2     = np.concatenate(all_cv2);     del all_cv2
    window  = np.concatenate(all_window);  del all_window
    step    = np.concatenate(all_step);    del all_step
    replica = np.concatenate(all_replica); del all_replica
    boost     = np.concatenate(all_boost);     del all_boost
    boost_dih = np.concatenate(all_boost_dih); del all_boost_dih
    pot_arr = np.concatenate(all_potential); del all_potential
    potential = pot_arr if np.any(np.isfinite(pot_arr)) else None
    u_nk    = np.concatenate(all_unk_blocks, axis=0); del all_unk_blocks

    epoch_src = np.concatenate(all_epoch_src) if all_epoch_src else np.zeros(len(cv), dtype=np.int32)
    del all_epoch_src

    # Drop per-state burnin frames: for each sample, compare its epoch-local step
    # against the burnin threshold for the state it was collected in.
    if np.any(burnin_by_k > 0):
        keep = step >= burnin_by_k[window]
        n_dropped = int((~keep).sum())
        if n_dropped > 0:
            print(f'    [burnin filter] dropped {n_dropped}/{len(cv)} samples ({100*n_dropped/len(cv):.1f}%) from pre-equilibration steps')
        cv = cv[keep]; cv2 = cv2[keep]; window = window[keep]
        step = step[keep]; replica = replica[keep]; boost = boost[keep]
        boost_dih = boost_dih[keep]
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_loaders_union_parquet.py -v`
Expected: PASS (5 tests). `test_load_parquet_adaptive_union_end_to_end_with_real_parquet_fixture` is the one that actually reaches both lazy `from analyze_gareus_mbar import ...` lines (in `_load_epoch_task` and in `load_parquet_adaptive_union` itself) — at this point in the plan, `analyze_gareus_mbar.py` still defines all four bias-reconstruction functions itself (Plan A3 relocates them later), so both imports resolve correctly. This is the real proof the lazy-import design works end-to-end, not just that the early-exit error paths work. Full pre-existing regression coverage (`tests/test_perf_loading_and_solver_memory.py`, `tests/test_union_mbar_per_epoch_bias.py`) is re-run against the new location in Task 9.

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/loaders_union_parquet.py tests/test_mbar_analysis_loaders_union_parquet.py
git commit -m "feat: add gareus.mbar_analysis.loaders_union_parquet (primary adaptive-production loader)"
```

---

### Task 8: Create `gareus/mbar_analysis/loaders.py` — single-run loaders + `prod_dir_of` + `load_data`

**Files:**
- Create: `gareus/mbar_analysis/loaders.py`
- Test: `tests/test_mbar_analysis_loaders.py` (new file)

**Interfaces:**
- Consumes: `Data`, `clean`, `infer_temp_beta`, `rjson`, `read_windows`, `jvec` (Task 1-2); `_find_adaptive_epoch_dirs`, `_find_selfcontained_epoch_dirs`, `_find_adaptive_epoch_csv_sources`, `_has_epoch_csv_layout`, `load_epoch_csv_adaptive`, `load_union_npz`, `_find_gareus_round_dirs`, `_augment_with_adaptive_rounds` (Task 5-6); `load_parquet_adaptive_union` (Task 7); `gareus.query.load_samples`/`.load_windows`/`.reconstruct_bias_matrix` (pre-existing, unchanged, imported lazily inside `load_parquet` exactly as it is today).
- Produces: `gareus.mbar_analysis.loaders.load_npz`, `.load_csv`, `.load_parquet`, `.prod_dir_of`, `.load_data` (the top-level dispatcher) — consumed directly by `analyze_gareus_mbar.py`'s `analyze()`/`main()` (unchanged, via re-export) and, per the spec, by no other in-scope plan directly (Plan A4 needs `_sample_block_ids` from `data.py`, not from this module).

- [ ] **Step 1: Write the failing test**

Create `tests/test_mbar_analysis_loaders.py`:

```python
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from gareus.mbar_analysis.loaders import (
    _npz_sample_count_open, _npz_window_count_open, _Arrays, _ANALYSIS_VECTOR_KEYS,
    _discover_analysis_chunk_paths, _append_npz_arrays, _load_merged_arrays, load_npz,
    _load_secondary_cv_from_csv, _csv_row_count_fast, _npz_sample_count,
    _analysis_binary_sample_count, _window_float_array, load_csv,
    _parquet_sample_count, prod_dir_of, load_data,
)


def _write_umbrella_windows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["center_A", "k_kcal_mol_A2"]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def test_npz_sample_count_open_prefers_step_key(tmp_path):
    npz_path = tmp_path / "a.npz"
    np.savez(npz_path, step=np.array([0, 1, 2]))
    with np.load(npz_path) as f:
        assert _npz_sample_count_open(f) == 3


def test_arrays_wrapper_exposes_files_property():
    a = _Arrays({"cv_A": np.array([1.0])})
    assert a.files == ["cv_A"]


def test_load_merged_arrays_and_load_npz_roundtrip(tmp_path):
    prod = tmp_path
    n = 3
    np.savez(
        prod / "analysis_arrays.npz",
        cv_A=np.arange(n, dtype=float), window=np.zeros(n, int),
        replica=np.zeros(n, int), step=np.arange(n),
        umbrella_reduced_bias_nk=np.zeros((n, 1)),
    )
    _write_umbrella_windows(prod / "umbrella_windows.csv", [{"center_A": "1.0", "k_kcal_mol_A2": "10.0"}])
    arr, notes = _load_merged_arrays(prod)
    assert arr["cv_A"].size == n
    d = load_npz(prod)
    assert d.cv.size == n
    assert d.centers.size == 1


def test_load_secondary_cv_from_csv_returns_nan_on_size_mismatch(tmp_path):
    p = tmp_path / "samples.csv"
    p.write_text("secondary_cv\n1.0\n2.0\n")
    out = _load_secondary_cv_from_csv(p, expected_size=5)
    assert out.size == 5
    assert np.all(np.isnan(out))


def test_csv_row_count_fast_counts_data_rows_not_header(tmp_path):
    p = tmp_path / "samples.csv"
    p.write_text("cv_A\n1.0\n2.0\n3.0\n")
    assert _csv_row_count_fast(p) == 3


def test_window_float_array_falls_back_to_default_on_bad_value():
    rows = [{"k": "1.5"}, {"k": "not-a-number"}]
    out = _window_float_array(rows, ("k",), default=-1.0)
    assert list(out) == [1.5, -1.0]


def test_load_csv_reconstructs_bias_matrix_from_windows(tmp_path):
    prod = tmp_path
    _write_umbrella_windows(prod / "umbrella_windows.csv", [
        {"center_A": "1.0", "k_kcal_mol_A2": "10.0"},
        {"center_A": "2.0", "k_kcal_mol_A2": "10.0"},
    ])
    with (prod / "samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cv_A", "window", "replica", "step"])
        w.writeheader()
        w.writerow({"cv_A": "1.0", "window": "0", "replica": "0", "step": "0"})
        w.writerow({"cv_A": "1.5", "window": "0", "replica": "0", "step": "1"})
    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_csv(prod)
    assert d.cv.size == 2
    assert d.u_nk.shape == (2, 2)


def test_parquet_sample_count_returns_zero_without_duckdb_data(tmp_path):
    assert _parquet_sample_count(tmp_path) == 0


def test_prod_dir_of_resolves_flat_npz_layout(tmp_path):
    (tmp_path / "analysis_arrays.npz").write_bytes(b"")
    assert prod_dir_of(tmp_path) == tmp_path.resolve()


def test_prod_dir_of_raises_when_nothing_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        prod_dir_of(tmp_path)


def test_load_data_auto_selects_csv_when_only_csv_present(tmp_path):
    prod = tmp_path
    _write_umbrella_windows(prod / "umbrella_windows.csv", [{"center_A": "1.0", "k_kcal_mol_A2": "10.0"}])
    with (prod / "samples.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["cv_A", "window", "replica", "step"])
        w.writeheader()
        w.writerow({"cv_A": "1.0", "window": "0", "replica": "0", "step": "0"})
    (prod / "run_args.json").write_text(json.dumps({"temperature_k": 300.0}))
    d = load_data(prod, out=None)
    assert d.cv.size == 1
    assert d.source.endswith("samples.csv")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mbar_analysis_loaders.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'gareus.mbar_analysis.loaders'`

- [ ] **Step 3: Write minimal implementation**

Create `gareus/mbar_analysis/loaders.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mbar_analysis_loaders.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Commit**

```bash
git add gareus/mbar_analysis/loaders.py tests/test_mbar_analysis_loaders.py
git commit -m "feat: add gareus.mbar_analysis.loaders (NPZ/CSV/Parquet single-run loaders + load_data dispatcher)"
```

---

### Task 9: Rewire `analyze_gareus_mbar.py` for `loaders_adaptive`/`loaders_union_parquet`/`loaders`

**Files:**
- Modify: `analyze_gareus_mbar.py` (delete 36 relocated definitions plus `_load_epoch_task`, which moved but is not re-exported — 37 total; add three import blocks)
- Test: `tests/test_analyze_gareus_mbar_uses_mbar_analysis_loaders.py` (new file)

**Interfaces:**
- Consumes: everything produced by Tasks 5-8.
- Produces: `analyze_gareus_mbar.{_find_adaptive_epoch_dirs, _find_selfcontained_epoch_dirs, _find_adaptive_epoch_csv_sources, _has_epoch_csv_layout, _find_adaptive_final_run_dirs, _find_gareus_round_dirs, _vectorized_map_lookup, _vectorized_map_lookup_or_self, _vectorized_map_index, load_epoch_csv_adaptive, load_union_npz, _load_round_raw, _build_union_window_table, _round_window_to_union_map, _augment_with_adaptive_rounds, _is_usable_for_mbar, _merge_missing_usable_states, load_parquet_adaptive_union, _npz_sample_count_open, _npz_window_count_open, _Arrays, _ANALYSIS_VECTOR_KEYS, _discover_analysis_chunk_paths, _append_npz_arrays, _load_merged_arrays, load_npz, _load_secondary_cv_from_csv, _csv_row_count_fast, _npz_sample_count, _analysis_binary_sample_count, _window_float_array, load_csv, _parquet_sample_count, load_parquet, prod_dir_of, load_data}` still exist as module attributes (unchanged names, now imported).

- [ ] **Step 1: Write the failing test**

Create `tests/test_analyze_gareus_mbar_uses_mbar_analysis_loaders.py`:

```python
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gareus.mbar_analysis.loaders_adaptive as mladapt
import gareus.mbar_analysis.loaders_union_parquet as mlunion
import gareus.mbar_analysis.loaders as mloaders
import analyze_gareus_mbar as agm

_ADAPTIVE_NAMES = [
    "_find_adaptive_epoch_dirs", "_find_selfcontained_epoch_dirs",
    "_find_adaptive_epoch_csv_sources", "_has_epoch_csv_layout",
    "_find_adaptive_final_run_dirs", "_find_gareus_round_dirs",
    "_vectorized_map_lookup", "_vectorized_map_lookup_or_self", "_vectorized_map_index",
    "load_epoch_csv_adaptive", "load_union_npz", "_load_round_raw",
    "_build_union_window_table", "_round_window_to_union_map",
    "_augment_with_adaptive_rounds",
]
_UNION_NAMES = ["_is_usable_for_mbar", "_merge_missing_usable_states", "load_parquet_adaptive_union"]
_LOADERS_NAMES = [
    "_npz_sample_count_open", "_npz_window_count_open", "_Arrays", "_ANALYSIS_VECTOR_KEYS",
    "_discover_analysis_chunk_paths", "_append_npz_arrays", "_load_merged_arrays", "load_npz",
    "_load_secondary_cv_from_csv", "_csv_row_count_fast", "_npz_sample_count",
    "_analysis_binary_sample_count", "_window_float_array", "load_csv",
    "_parquet_sample_count", "load_parquet", "prod_dir_of", "load_data",
]


def test_adaptive_names_are_shared_objects():
    for name in _ADAPTIVE_NAMES:
        assert getattr(agm, name) is getattr(mladapt, name), name


def test_union_names_are_shared_objects():
    for name in _UNION_NAMES:
        assert getattr(agm, name) is getattr(mlunion, name), name


def test_loaders_names_are_shared_objects():
    for name in _LOADERS_NAMES:
        assert getattr(agm, name) is getattr(mloaders, name), name


def test_no_circular_import_via_fresh_subprocess():
    import subprocess
    result = subprocess.run(
        [sys.executable, "-c", "import gareus.mbar_analysis.loaders"],
        cwd=str(Path(__file__).resolve().parent.parent),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analyze_gareus_mbar_uses_mbar_analysis_loaders.py -v`
Expected: FAIL — every `is` assertion in the first three tests fails (local definitions still present); the fourth test passes already (it doesn't depend on `analyze_gareus_mbar.py`'s own state).

- [ ] **Step 3: Write minimal implementation**

In `analyze_gareus_mbar.py`, immediately after the `gareus.mbar_analysis.data` import block Task 4 added, insert:

```python
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
```

Then delete each of the 36 original definitions this replaces
(`_find_adaptive_epoch_dirs` through `load_data`, per the grep anchor
below) from their current locations further down the file. Re-confirm exact
ranges before editing (line numbers have drifted since this plan was
written):

```bash
grep -n "^def _find_adaptive_epoch_dirs\|^def _find_selfcontained_epoch_dirs\|^def _find_adaptive_epoch_csv_sources\|^def _has_epoch_csv_layout\|^def _find_adaptive_final_run_dirs\|^def _find_gareus_round_dirs\|^def _vectorized_map_lookup\|^def _vectorized_map_lookup_or_self\|^def _vectorized_map_index\|^def load_epoch_csv_adaptive\|^def load_union_npz\|^def _load_round_raw\|^def _build_union_window_table\|^def _round_window_to_union_map\|^def _augment_with_adaptive_rounds\|^def _is_usable_for_mbar\|^def _merge_missing_usable_states\|^def _load_epoch_task\|^def load_parquet_adaptive_union\|^def _npz_sample_count_open\|^def _npz_window_count_open\|^class _Arrays\|^_ANALYSIS_VECTOR_KEYS\|^def _discover_analysis_chunk_paths\|^def _append_npz_arrays\|^def _load_merged_arrays\|^def load_npz\|^def _load_secondary_cv_from_csv\|^def _csv_row_count_fast\|^def _npz_sample_count\|^def _analysis_binary_sample_count\|^def _window_float_array\|^def load_csv\|^def _parquet_sample_count\|^def load_parquet\|^def prod_dir_of\|^def load_data" analyze_gareus_mbar.py
```

**Important**: do **not** delete `_compute_u_nk_analytical`,
`_parse_epoch_window_map_native_params`, `_epoch_bias_param_vectors`, or
`_reconstruct_union_bias_block` — these four stay in
`analyze_gareus_mbar.py` for Plan A3 (see Global Constraints and the
design spec's Recommended Approach). Do not delete `_load_epoch_task`
either — it moved to `loaders_union_parquet.py` in Task 7, but grep
confirms (per the design spec) it is referenced nowhere else in this file,
so it is one of the two names Task 4/9 deliberately does not re-export;
its *definition*, however, must still be deleted here since it now lives
in `loaders_union_parquet.py`. Do not reformat or reorder anything else.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analyze_gareus_mbar_uses_mbar_analysis_loaders.py -v`
Expected: PASS (4 tests, 36 assertions across the three identity tests)

- [ ] **Step 5: Confirm the rest of the file still works**

Run: `python -m py_compile analyze_gareus_mbar.py`
Expected: clean compile.

Run:
```bash
pytest -q tests/test_prod_dir_resolution.py tests/test_mbar_epoch_filter.py \
  tests/test_perf_loading_and_solver_memory.py tests/test_final_registry_merge.py \
  tests/test_union_mbar_per_epoch_bias.py tests/test_secondary_cv_regime_split.py \
  tests/test_epoch0_pmf_gamd_split.py tests/test_masked_logw_subset_pmf.py \
  tests/test_bias_reconstruction_nan_handling.py -v
```
Expected: PASS, identical results to before this task — these are the
existing test files that exercise this domain's relocated names (plus, for
the last three, names Plan A3 will relocate) through `analyze_gareus_mbar`'s
namespace; per the Global Constraints, none of them need code changes.

Run: `python -c "import gareus.mbar_analysis.loaders"`
Expected: no exception — the real circular-import smoke test traced in the
design spec's Architecture section (this single import transitively pulls
in `loaders.py` → `loaders_union_parquet.py` → `loaders_adaptive.py` →
`data.py`; if either lazy reverse-import into `analyze_gareus_mbar` were
accidentally placed at module top, this would fail here).

Run: `pytest -q tests/ --ignore=tests/test_validation_common.py -k "not physics_oracle and not thermodynamic_validity"`
Expected: same pre-existing failures as before this task, no new ones.

- [ ] **Step 6: Commit**

```bash
git add analyze_gareus_mbar.py tests/test_analyze_gareus_mbar_uses_mbar_analysis_loaders.py
git commit -m "refactor: import loader-domain names from gareus.mbar_analysis.{loaders_adaptive,loaders_union_parquet,loaders} instead of redefining"
```

---

### Task 10: Full-repo regression pass + plan self-review

**Files:**
- None (verification only; fixes here address anything the checks below surface).

**Interfaces:** None new.

- [ ] **Step 1: Full test suite**

Run: `pytest -q tests/ --ignore=tests/test_validation_common.py`
Expected: same result as on `main` before this plan (the 6 pre-existing, unrelated failures documented in Plan A1's own Testing section — help-text encyclopedia numbering, threadpool-import check, missing validation launcher fixtures — plus `test_validation_common.py`'s own pre-existing `ModuleNotFoundError`, excluded the same way A1 excludes it). No new failures.

- [ ] **Step 2: Compile check across every touched file**

Run: `python -m py_compile analyze_gareus_mbar.py gareus/mbar_analysis/data.py gareus/mbar_analysis/loaders_adaptive.py gareus/mbar_analysis/loaders_union_parquet.py gareus/mbar_analysis/loaders.py`
Expected: clean.

- [ ] **Step 3: `git diff --check` for whitespace errors**

Run: `git diff --check`
Expected: no output.

- [ ] **Step 4: End-to-end smoke test against a real run directory**

Run `python analyze_gareus_mbar.py <a real completed run's output dir> --out /tmp/a2_smoke_test` (or, if none is available in this environment, construct a minimal synthetic `samples.csv` + `umbrella_windows.csv` run directory as in Task 8's `test_load_data_auto_selects_csv_when_only_csv_present`).
Expected: runs to completion, `pmf_summary.json` written, no new warnings about missing names.

- [ ] **Step 5: Plan self-review**

- **Spec coverage**: every function/class named in the design spec's Problem
  and Architecture sections has a corresponding Task above (`Data` — Task 1;
  `rjson`/`wjson`/`_json_default`/`read_windows`/`jvec` — Task 1;
  `infer_temp_beta`/`clean`/`_masked_data`/`_apply_analysis_stride`/
  `_filter_epoch_source`/`_sample_block_ids`/`_skip_first_n_frames` — Task 2;
  `_epoch_dir_index`/`_epoch_number_for_run_dir`/
  `_epoch_run_manifest_secondary_cv_type`/`_epoch_zero_split_masks`/
  `_secondary_cv_epoch_regime_masks` — Task 3; adaptive-production discovery
  + vectorized lookup — Task 5; `load_epoch_csv_adaptive`/`load_union_npz`/
  rounds augmentation — Task 6; `load_parquet_adaptive_union` cluster —
  Task 7; NPZ/CSV/Parquet single-run loaders + `prod_dir_of`/`load_data` —
  Task 8). The four bias-reconstruction functions and the
  plotting/convergence/trajectory helpers listed in the spec's Out Of Scope
  are correctly absent from every task.
- **Placeholder scan**: every task's Step 3 contains complete function
  bodies copied verbatim from the current source, not a description of what
  the code should do; every RED test in Step 1 is a real, runnable pytest
  file with concrete assertions (no `# TODO: add appropriate tests`).
- **Type/signature consistency**: every relocated function keeps its exact
  original signature (verified by direct comparison against the source read
  during spec research); the two lazy-import call sites
  (`_augment_with_adaptive_rounds` in Task 6, `_load_epoch_task`/
  `load_parquet_adaptive_union` in Task 7) import from `analyze_gareus_mbar`
  by the same names those four functions have today, so Plan A3 only needs
  to change the module path in those two spots, not any call-site argument
  list.
- **Dependency order check**: Task 1-3 (data.py) precede Task 4 (data.py
  rewire) precede Task 5-8 (the three loader modules, which all import from
  `data.py`) precede Task 9 (loaders rewire) — matches the acyclic
  dependency graph in the design spec exactly; no task imports from a module
  a later task creates.
- **No static linter available**: addressed by requiring real invocation
  (not just import) of every moved function inside each module-creation
  task's own RED test, per Global Constraints — confirmed present in every
  one of Tasks 1, 2, 3, 5, 6, 7, 8's test files above (each calls, not just
  imports, every new public name at least once).

