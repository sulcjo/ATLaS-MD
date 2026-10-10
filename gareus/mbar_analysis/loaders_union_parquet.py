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
import shutil
import tempfile
import atexit
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np

from gareus.units import K_B_KJ_PER_MOL_K
from .data import (Data, clean, infer_temp_beta, rjson, _fill_masked_nan,
                   analysis_stride_keep_mask,
                   _epoch_run_manifest_secondary_cv_type)
from .loaders_adaptive import (_find_adaptive_epoch_dirs, _phase_label,
                               _validate_and_repair_epoch_window_map,
                               _vectorized_map_lookup, _vectorized_map_index)
from .bias import _reconstruct_union_bias_block_per_regime
from .cv2_reprojection import (CV2_REPROJECTION_FILENAME,
                               CV2_REPROJECTION_MIN_COVERAGE, join_on_step)
from .storage import create_matrix, row_blocks, release_pages, select_matrix, cleanup_on_error


def _is_usable_for_mbar(row: dict) -> bool:
    return str(row.get('usable_for_mbar', '')).strip().lower() in ('true', '1', 'yes')


def _spool_u_nk_blocks(blocks: list, directory: Path):
    """Merge bias blocks into disk-backed storage without a second RAM copy."""
    if not blocks:
        raise ValueError('cannot spool empty u_nk block list')
    first = np.asarray(blocks[0])
    if first.ndim != 2:
        raise ValueError(f'u_nk block must be 2-D, got shape {first.shape}')
    rows = sum(int(np.asarray(block).shape[0]) for block in blocks)
    columns = int(first.shape[1])
    matrix = create_matrix((rows, columns), directory=directory)
    path = Path(matrix.filename)
    offset = 0
    try:
        for block in blocks:
            block = np.asarray(block)
            if block.ndim != 2 or block.shape[1] != columns:
                raise ValueError('u_nk blocks have inconsistent shapes')
            end = offset + block.shape[0]
            for start, stop in row_blocks(*block.shape):
                matrix[offset + start:offset + stop] = block[start:stop]
                release_pages(matrix, written=True)
                release_pages(block)
            offset = end
        matrix.flush()
    except Exception:
        del matrix
        path.unlink(missing_ok=True)
        raise
    return matrix, path


def _cleanup_memmap_file(path: Path) -> None:
    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass


def _materialize_block_files(block_paths: list[Path], rows: int, columns: int,
                             directory: Path) -> tuple[np.memmap, Path]:
    """Combine disk-backed per-epoch blocks without retaining them in RAM."""
    matrix = create_matrix((rows, columns), directory=directory)
    path = Path(matrix.filename)
    offset = 0
    try:
        for block_path in block_paths:
            block = np.load(block_path, mmap_mode='r', allow_pickle=False)
            end = offset + int(block.shape[0])
            for start, stop in row_blocks(*block.shape):
                matrix[offset + start:offset + stop] = block[start:stop]
                release_pages(matrix, written=True)
                release_pages(block)
            offset = end
            del block
        matrix.flush()
    except Exception:
        del matrix
        _cleanup_memmap_file(path)
        raise
    for block_path in block_paths:
        _cleanup_memmap_file(block_path)
    atexit.register(_cleanup_memmap_file, path)
    return matrix, path


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
    # NOTE: _parse_epoch_window_map_native_params now lives in
    # gareus.mbar_analysis.bias (Plan A3) and analyze_gareus_mbar.py
    # re-exports it unchanged -- importing it from analyze_gareus_mbar here
    # (rather than switching to gareus.mbar_analysis.bias directly) is
    # deliberate, not stale: this import MUST stay function-body-local
    # (lazy) for the same circular-import reason documented in
    # loaders_adaptive.py's _augment_with_adaptive_rounds.
    from analyze_gareus_mbar import _parse_epoch_window_map_native_params
    with wmap_path.open(newline='') as f:
        wmap_rows = list(csv.DictReader(f))
    samples = load_samples(epoch_dir, n_threads=n_threads)
    # The map's row count used to be trusted blindly. It can be STALE: an
    # `--us-auto-drop-bad-windows` phase renumbers its surviving windows
    # 0..N-1 without ever rewriting the map the registry already wrote over
    # all *active* states, so every local index at or after the first dropped
    # one resolves to the wrong state (~15% of a real 12.1M-sample campaign,
    # docs/chignolin_6_low_ess_root_cause.md). Repair it in memory where the
    # phase's own drop record allows, fail closed where it does not -- and do
    # it *before* wmap/native_params are derived, so both follow the repaired
    # rows. Note the samples must already be loaded above: the row-count check
    # that needs no metadata at all compares against the window_id values the
    # Parquet data actually contains.
    wmap_rows, wmap_notes = _validate_and_repair_epoch_window_map(
        epoch_dir, wmap_rows,
        window_ids=samples.get('window_id') if samples else None,
        cv2=_fill_masked_nan(samples.get('cv2')) if samples else None)
    wmap = {int(r['epoch_window']): int(r['state_id']) for r in wmap_rows}
    native_params = _parse_epoch_window_map_native_params(wmap_rows)
    ep_meta = rjson(epoch_dir / 'umbrella_pymbar_metadata.json', {})
    return samples, wmap, ep_meta, native_params, wmap_notes


def _secondary_cv_regime_change_note(epoch_dirs: list, sample_counts: Optional[list] = None) -> Optional[str]:
    """Warn when the secondary CV was *redefined* part-way through a campaign.

    Each epoch/phase records the secondary-CV mode it actually ran under in
    its own ``run_manifest.json`` (``resolved_args.secondary_cv``). Two
    different modes -- e.g. ``torsion-pca`` for epoch 0 and ``tica-linear``
    from the tICA CV2 auto-switch onwards -- are different linear projections
    of the same raw torsion features, i.e. genuinely different order
    parameters, not a recentering of one coordinate. The per-epoch native
    window params keep each epoch's own ``u_nk`` internally consistent (the
    2026-08-04 fix), but state *k* is then not one Hamiltonian across the
    switch, which is formally invalid for the single pooled MBAR solve that
    spans both.

    Returns None for the overwhelming majority of runs (one regime, or no
    readable manifests). This only reports; restructuring MBAR itself is a
    much larger question. CV2-*facing plots* are already split per regime by
    ``_secondary_cv_epoch_regime_masks``; nothing surfaced the MBAR-side
    implication before. Deliberately reads only each phase dir's own manifest
    (no parent fallback), exactly like that sibling function, so the two can
    never disagree about how many regimes a run has.
    """
    regimes: dict = {}
    for i, ed in enumerate(epoch_dirs):
        regime = _epoch_run_manifest_secondary_cv_type(Path(ed))
        if not regime:
            continue
        entry = regimes.setdefault(regime, {'phases': [], 'samples': 0})
        entry['phases'].append(_phase_label(ed))
        if sample_counts is not None and i < len(sample_counts):
            entry['samples'] += int(sample_counts[i])
    if len(regimes) < 2:
        return None
    # Reported in first-appearance order, which is the epoch-dir discovery
    # order (sorted, hence chronological for the epoch_NNN/final layout).
    parts = []
    for regime, e in regimes.items():
        # A late-epoch regime can span a dozen topup sub-runs; list a few and
        # count the rest so the warning stays readable in pmf_summary.json.
        shown = e['phases'][:4]
        phases = ', '.join(shown)
        if len(e['phases']) > len(shown):
            phases += f' (+{len(e["phases"]) - len(shown)} more)'
        parts.append(f'{regime!r} ({phases}; {e["samples"]:,} samples)')
    return ('[cv2 regime change] The secondary CV was redefined mid-campaign: '
            + ' then '.join(parts)
            + ', per each phase\'s own run_manifest.json (resolved_args.secondary_cv). '
              'Bias energies use each epoch\'s own native window params, so u_nk is internally '
              'consistent *within* each regime -- but state k is NOT one Hamiltonian across the '
              'switch, so this single pooled MBAR solve is formally invalid across the regime '
              'boundary: treat cross-regime free-energy differences, and any pooled CV2-facing '
              'number, as unvalidated. CV2-facing plots are already split per regime; the MBAR '
              'solve itself is not.')


def _load_cv2_reprojection(adaptive_dir: Path) -> Optional[dict]:
    """Read ``cv2_reprojected.parquet`` if a run has one.

    Written by ``reproject_cv2.py``; absent unless somebody generated it. It
    holds, per regime, the value that regime's CV2 definition takes on each
    stored frame -- which is what lets every u_nk column be evaluated in its
    own definition instead of whichever one happened to be in effect when the
    row was recorded.

    Returns ``{regime: {'steps': int64[], 'cv2': float64[]}}``, or None when
    the file is missing or unreadable. Unreadable is treated as missing: the
    caller falls back to the single-cv2 path and warns, which is strictly
    better than aborting an analysis over an optional side-car.
    """
    path = Path(adaptive_dir) / CV2_REPROJECTION_FILENAME
    if not path.is_file():
        return None
    try:
        import pyarrow.parquet as pq
        table = pq.read_table(str(path))
    except Exception:
        return None
    names = list(table.schema.names)
    if 'steps' not in names:
        return None
    steps = np.asarray(table['steps'], dtype=np.int64)
    out: dict = {}
    for name in names:
        if not name.startswith('cv2_'):
            continue
        vals = np.asarray(table[name], dtype=np.float64)
        finite = np.isfinite(vals)
        out[name[len('cv2_'):]] = {'steps': steps[finite], 'cv2': vals[finite]}
    return out or None


def _per_regime_bias_block(cv_epoch, step_epoch, beta, pc_e, pk_e, sc_e, sk_e,
                            state_regimes, reproj: dict, cv2_epoch, epoch_regime: str,
                            low_memory: bool = False, directory=None):
    """One epoch-block's u_nk with each column in its own CV2 definition.

    Returns ``(block, coverage)``, or ``(None, {})`` when the table cannot
    supply every regime this block's columns need -- in which case the caller
    keeps the existing single-cv2 behaviour rather than silently filling gaps.

    The epoch's OWN regime does not come from the table: ``cv2_epoch`` is the
    value that was actually recorded while these frames ran, so it is exact
    for every row, whereas the table only covers rows whose step was
    recoverable. Using the recorded values where they exist keeps coverage as
    high as possible and avoids a needless round-trip through a reprojection
    of a CV we already have.
    """
    needed = list(dict.fromkeys(state_regimes))
    cv2_by_regime: dict = {}
    coverage: dict = {}
    for regime in needed:
        if regime and regime == epoch_regime:
            cv2_by_regime[regime] = np.asarray(cv2_epoch, dtype=np.float64)
            coverage[regime] = 1.0
            continue
        entry = reproj.get(regime)
        if entry is None:
            return None, {}
        vals, matched = join_on_step(step_epoch, entry['steps'], entry['cv2'])
        cov = float(matched.mean()) if matched.size else 0.0
        if cov < CV2_REPROJECTION_MIN_COVERAGE:
            # Decline rather than degrade. An uncovered row gets NaN in this
            # regime's u_nk columns and is then DROPPED by clean(), so a sparse
            # table does not blur the result -- it deletes the samples. Keeping
            # every sample under the reported splice beats losing most of a regime.
            return None, {'declined_regime': regime, 'declined_coverage': cov,
                          'required': CV2_REPROJECTION_MIN_COVERAGE,
                          'rows': int(matched.size),
                          'would_drop': int(matched.size - int(matched.sum()))}
        cv2_by_regime[regime] = vals
        coverage[regime] = cov
    block = _reconstruct_union_bias_block_per_regime(
        cv_epoch, cv2_by_regime, beta, pc_e, pk_e, sc_e, sk_e, state_regimes,
        low_memory=low_memory, directory=directory)
    return block, coverage


def _aux_phase_label(phase_dir, adaptive_dir) -> str:
    """Phase label relative to the campaign (``epoch_001``, ``epoch_001/baseline``, ``final``)."""
    try:
        return Path(phase_dir).resolve().relative_to(Path(adaptive_dir).resolve()).as_posix()
    except ValueError:
        return Path(phase_dir).name


@cleanup_on_error
def load_parquet_adaptive_union(adaptive_dir: Path, n_threads: int = 0, n_workers: int = 4,
                                epoch_ids: Optional[set[int]] = None,
                                low_memory: bool = False, analysis_stride: int = 1,
                                analysis_stride_offset: int = 0, memory_reporter=None) -> Data:
    """Load MBAR inputs from adaptive-production Parquet epoch data.

    Pools samples from all epoch run directories, remaps per-epoch window IDs to
    global state IDs via epoch_window_map.csv, and builds the union N×K bias
    matrix against the full registry of usable states.

    n_threads: DuckDB threads per connection (0=auto, capped at min(cpu_count,64))
    n_workers: parallel epoch-dir workers; each opens its own DuckDB connection
    low_memory: use one epoch worker at a time instead of retaining all worker
        results; callers should prefer the resolved union NPZ when available.
    """
    from gareus.kernel_identity import aux_admission_allows_pooling, refuse_aux_snapshots
    # A campaign whose driver admitted aux workers pools them (per-sample z, aux term); otherwise refuse as before.
    _aux_rec = aux_admission_allows_pooling(Path(adaptive_dir))
    if _aux_rec is None:
        refuse_aux_snapshots(Path(adaptive_dir), where="adaptive union loader")
    try:
        from gareus.query import load_samples  # noqa: F401 – used in _load_epoch_task
    except ImportError as exc:
        raise ImportError(f'gareus package required for Parquet loading: {exc}') from exc
    # NOTE: both of these now live in gareus.mbar_analysis.bias (Plan A3);
    # see _load_epoch_task's own note above for why importing them from
    # analyze_gareus_mbar's re-export (rather than from gareus.mbar_analysis.bias
    # directly) is deliberate, and why this import must stay lazy.
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
    # lambda-ladder rung per state, from state_registry.csv's own gamd_lambda
    # column (WindowStateRegistry.write_state_csv) -- 0.0 (ladder inactive)
    # when the column is absent (older registry snapshot) or blank.
    state_lambdas  = np.array([float(r.get('gamd_lambda', 0.0) or 0.0) for r in reg_rows], dtype=np.float64)

    epoch_dirs = _find_adaptive_epoch_dirs(adaptive_dir, epoch_ids=epoch_ids)
    if not epoch_dirs:
        selected = f' for requested epoch(s) {sorted(epoch_ids)}' if epoch_ids is not None else ''
        raise FileNotFoundError(f'No epoch Parquet data found in {adaptive_dir}{selected}')
    # Final fix wave M3: every phase directory whose samples are read is checked with run_has_aux (run
    # manifest, snapshots, samples payload; pure JSON) on top of the depth-2 snapshot scan above.
    from gareus.kernel_identity import refuse_aux_run
    if _aux_rec is None:
        for _ed, _ in epoch_dirs:
            refuse_aux_run(Path(_ed), where="adaptive union loader")
    # Admitted aux workers {column k: aux params}, read from the live registry (the CSV carries no metadata).
    _aux_cols: dict = {}
    if _aux_rec is not None:
        from gareus.adaptive import aux_pooling as _ap
        from gareus.adaptive_production import WindowStateRegistry
        from gareus.kernel_identity import AuxPoolingRefused
        if (_aux_rec.get('workers') or []) and not (Path(adaptive_dir) / 'state_registry.json').exists():
            raise AuxPoolingRefused(f'{adaptive_dir}: aux_admission.json lists workers but state_registry.json is missing')
        _aux_reg = WindowStateRegistry.load(adaptive_dir)
        _aux_by_sid = _ap.worker_table((int(st.state_id), st.metadata) for st in _aux_reg.all_states())
        for _sid, _rec in _aux_by_sid.items():
            if _sid in state_id_to_k:
                if str(_rec.get('aux_model_sha256')) != str(_aux_rec.get('model_sha256')):
                    from gareus.kernel_identity import AuxPoolingRefused
                    raise AuxPoolingRefused(f'worker state {_sid} names aux model {str(_rec.get("aux_model_sha256"))[:12]}, '
                                            f'admission record {str(_aux_rec.get("model_sha256"))[:12]}')
                _aux_cols[state_id_to_k[_sid]] = _rec
        _ap.require_admitted_workers(_aux_rec, _aux_by_sid, 'adaptive union loader',
                                     pooled={int(state_ids[k]): r for k, r in _aux_cols.items()})
    if _aux_rec is not None:
        # Admitted campaign: a model-run segment the eligibility rule would drop (aux_unpersisted) refuses.
        from gareus.adaptive import aux_pooling as _ap
        for _ed, _ in epoch_dirs:
            _ap.require_persisted_model_segments(Path(_ed), str(_aux_rec['model_sha256']),
                                                 _aux_phase_label(_ed, adaptive_dir))

    all_cv = []; all_cv2 = []; all_window = []; all_step = []
    all_replica = []; all_boost = []; all_boost_dih = []; all_potential = []; all_epoch_src = []
    all_v_pep = []; all_v_dih = []
    all_aux_z = []
    all_gamd_lambda_sample = []
    all_unk_blocks = []
    u_nk_block_paths: list[np.memmap] = []
    beta = float('nan')
    meta: dict = rjson(adaptive_dir.parent / 'run_manifest.json', {})

    # Compute per-connection thread budget: distribute n_threads across n_workers.
    _cpu_cap = min(os.cpu_count() or 64, int(os.environ.get('NUMEXPR_MAX_THREADS', 64)))
    _total_threads = n_threads if n_threads > 0 else _cpu_cap
    _n_workers = 1 if low_memory else min(len(epoch_dirs), max(1, n_workers))
    _threads_per_conn = max(1, _total_threads // _n_workers)

    # Load all epochs in parallel (I/O bound); post-process sequentially (order-dependent).
    # Low-memory mode keeps only the current worker result alive.
    if low_memory:
        epoch_iter = (
            (_load_epoch_task(_ed, _wp, _threads_per_conn), (_ed, _wp))
            for _ed, _wp in epoch_dirs
        )
    else:
        with ThreadPoolExecutor(max_workers=_n_workers) as _pool:
            epoch_loaded = list(_pool.map(
                lambda _ewt: _load_epoch_task(_ewt[0], _ewt[1], _ewt[2]),
                [(ed, wp, _threads_per_conn) for ed, wp in epoch_dirs],
            ))
        epoch_iter = zip(epoch_loaded, epoch_dirs)

    # Any phase whose stale window map had to be repaired in memory says so
    # here (an unrepairable one raised inside the worker instead). Printed
    # immediately *and* carried into meta['load_notes'], which analyze()
    # folds into pmf_summary.json's warnings -- a silent repair would be as
    # misleading as the bug it fixes.
    load_notes = []

    # Resolve beta once, up front, using the same fallback order as before (first
    # epoch whose metadata yields it, else a top-level adaptive_dir inference).
    # Must be fixed *before* any per-epoch bias block is built below, since every
    # block needs the same beta.
    for epoch_dir, _ in epoch_dirs:
        if math.isfinite(beta):
            break
        ep_meta = rjson(epoch_dir / 'umbrella_pymbar_metadata.json', {})
        b = float(ep_meta.get('beta_1_over_kJ_mol') or 0.0)
        beta = b if b > 0 else infer_temp_beta(epoch_dir, ep_meta)[1]
    if not math.isfinite(beta):
        _, beta = infer_temp_beta(adaptive_dir, meta)

    per_dir_valid_counts = [0] * len(epoch_dirs)
    # _epoch_source indexes the phases that CONTRIBUTED samples (skipped phases get no index), so it is
    # not an index into adaptive_epoch_run_dirs (every discovered phase) once any phase is skipped.
    epoch_source_dirs: list = []
    # Optional side-car giving every regime's CV2 for every recoverable frame
    # (reproject_cv2.py). Present only if somebody generated it; when absent
    # the pooled u_nk keeps its existing per-epoch-native cv2, which across a
    # CV2 regime change is a splice -- warned about below.
    _reproj = _load_cv2_reprojection(adaptive_dir)
    _reproj_coverage: dict = {}
    _reproj_declined: dict = {}
    _final_regime = (_epoch_run_manifest_secondary_cv_type(Path(epoch_dirs[-1][0]))
                     if epoch_dirs else '') or ''

    for _ei, ((samples, wmap, ep_meta, native_params, _notes), (epoch_dir, _)) in enumerate(epoch_iter):
        if memory_reporter is not None:
            memory_reporter.record(f'epoch_loaded:{_phase_label(epoch_dir)}')
        load_notes.extend(_notes)
        for _n in _notes:
            print(f'    {_n}')
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
        if low_memory and (analysis_stride > 1 or analysis_stride_offset > 0):
            valid_idx = np.flatnonzero(valid)
            stride_keep = analysis_stride_keep_mask(
                samples['replica'][valid_idx], samples['step'][valid_idx],
                np.full(valid_idx.size, _ei, dtype=np.int32),
                analysis_stride, analysis_stride_offset)
            valid[valid_idx[~stride_keep]] = False
            if not np.any(valid):
                continue
        # Filter first, cast second: avoids allocating a full-epoch-length
        # float64 transient that's then mostly discarded by [valid].
        cv_epoch = samples['cv1'][valid].astype(np.float64, copy=False)
        all_cv.append(cv_epoch)
        epoch_source_dirs.append(str(epoch_dir))  # _epoch_source value len(all_cv) - 1 -> this phase
        cv2_raw = samples.get('cv2')
        cv2_epoch = _fill_masked_nan(cv2_raw[valid]) if cv2_raw is not None else np.full(valid.sum(), np.nan)
        all_cv2.append(cv2_epoch)
        # Vectorized equivalent of [state_id_to_k[int(s)] for s in remapped[valid]]
        # -- see _vectorized_map_index docstring; raises KeyError on a missing
        # state_id exactly like the original dict subscripting did. Narrowed to
        # int16 (Data.window's on-disk source is uint16; downstream consumers
        # already defensively re-cast to int64 before use -- see CLAUDE.md/audit).
        all_window.append(_vectorized_map_index(remapped[valid], state_id_to_k, dtype=np.int16))
        step_epoch = samples['step'][valid].astype(np.int64, copy=False)
        all_step.append(step_epoch)
        # NOTE: intentionally NOT applying the filter-then-cast reorder here --
        # the `else` branch already builds an array sized to valid.sum() (not
        # the full epoch length), and rebasing it on `[valid]` after slicing
        # would require restructuring around a latent shape mismatch in that
        # branch (np.zeros(int(valid.sum()), ...) then indexed again by the
        # full-length `valid` mask) that is out of scope to touch here.
        rep = samples['replica'].astype(np.int16) if 'replica' in samples else np.zeros(int(valid.sum()), np.int16)
        all_replica.append(rep[valid])
        # Same masked-null hazard as cv2 above (and same fix): gamd_boost_total/
        # gamd_boost_dihedral are real SQL NULLs for every sample of a
        # non-GaMD/plain-umbrella run, and this module's own
        # _concat_numpy_dicts fix (gareus/query.py) means samples.get(...)
        # now correctly hands back a live-masked array for them instead of a
        # silently-denatured one -- a bare .astype()[valid] preserves that
        # mask (both .astype() and boolean indexing keep it), and the
        # unconditional np.concatenate(all_boost) below would then silently
        # drop it again, exposing the arbitrary fill value as fabricated real
        # boost/potential data. Route through _fill_masked_nan so every
        # per-epoch block is already a plain NaN-filled array before that
        # final concatenate.
        boost_raw = samples.get('gamd_boost_total')
        all_boost.append(_fill_masked_nan(boost_raw[valid]) if boost_raw is not None else np.full(valid.sum(), np.nan))
        boost_dih_raw = samples.get('gamd_boost_dihedral')
        all_boost_dih.append(_fill_masked_nan(boost_dih_raw[valid]) if boost_dih_raw is not None else np.full(valid.sum(), np.nan))
        # Same masked-null hazard/fix as boost/boost_dih above: lambda-ladder
        # raw channel energies are real SQL NULLs on any run/segment where
        # the ladder is not active or predates the schema addition.
        v_pep_raw = samples.get('v_pep_kj_mol')
        all_v_pep.append(_fill_masked_nan(v_pep_raw[valid]) if v_pep_raw is not None else np.full(valid.sum(), np.nan))
        v_dih_raw = samples.get('v_dih_kj_mol')
        all_v_dih.append(_fill_masked_nan(v_dih_raw[valid]) if v_dih_raw is not None else np.full(valid.sum(), np.nan))
        # Per-sample gamd_lambda (production.snapshot_window_rows / gareus/store.py) is
        # this loop's independent witness against state_registry.csv's own gamd_lambda
        # column (state_lambdas, built above from the registry) -- see
        # assert_lambda_sources_agree, called once below after this is concatenated.
        # Same masked-null hazard/fix as boost/v_pep/v_dih above.
        lam_sample_raw = samples.get('gamd_lambda')
        all_gamd_lambda_sample.append(_fill_masked_nan(lam_sample_raw[valid]) if lam_sample_raw is not None else np.full(valid.sum(), np.nan))
        pot_raw = samples.get('potential')
        all_potential.append(_fill_masked_nan(pot_raw[valid]) if pot_raw is not None else np.full(valid.sum(), np.nan))
        all_epoch_src.append(np.full(int(valid.sum()), len(all_cv) - 1, dtype=np.int32))
        per_dir_valid_counts[_ei] = int(valid.sum())
        # Bias energies for THIS epoch's samples must use the window params that
        # were actually in effect during this epoch (native_params), not whatever
        # a state's row in the live/final registry says today — that snapshot can
        # be stale for any state recentered by a later epoch (e.g. the tICA CV2
        # auto-switch overwrites secondary_center in place; see CLAUDE.md).
        pc_e, pk_e, sc_e, sk_e = _epoch_bias_param_vectors(
            native_params, state_ids, primary_centers, primary_ks, sec_centers, sec_ks)
        block = None
        if _reproj is not None:
            # Every column evaluated in the CV2 definition its own secondary
            # params were written for. The attribution follows directly from
            # how _epoch_bias_param_vectors just chose them: a state this
            # epoch's own snapshot covers took THIS epoch's params, so this
            # epoch's regime; any other state fell back to the final registry,
            # whose secondary params are whatever the last regime left there.
            _epoch_regime = _epoch_run_manifest_secondary_cv_type(Path(epoch_dir)) or ''
            _state_regimes = [
                (_epoch_regime if int(sid) in native_params else _final_regime)
                for sid in state_ids
            ]
            block, _cov = _per_regime_bias_block(
                cv_epoch, step_epoch, beta, pc_e, pk_e, sc_e, sk_e,
                _state_regimes, _reproj, cv2_epoch, _epoch_regime, low_memory=low_memory,
                directory=adaptive_dir)
            if block is not None:
                _reproj_coverage[str(epoch_dir)] = _cov
            elif _cov.get('declined_regime'):
                _reproj_declined[str(epoch_dir)] = _cov
        if block is None:
            block = _reconstruct_union_bias_block(cv_epoch, cv2_epoch, beta, pc_e, pk_e, sc_e, sk_e,
                                                  low_memory=low_memory, directory=adaptive_dir)
        if _aux_cols:
            # Worker restraint 0.5 k (z - c)^2 (kcal/mol -> kJ -> reduced) for every sample of the phase under
            # every worker column, added before the ladder boost (workers are lambda = 0).
            _recorded = samples.get(_ap.Z_COLUMN)
            _z = _ap.phase_z(_aux_phase_label(epoch_dir, adaptive_dir), epoch_dir, rep[valid], step_epoch,
                             str(_aux_rec['model_sha256']),
                             recorded=None if _recorded is None else _recorded[valid],
                             admission=_aux_rec, adaptive_dir=adaptive_dir)
            all_aux_z.append(np.asarray(_z, dtype=np.float64))
            for _k, _rec in _aux_cols.items():
                block[:, _k] += beta * _ap.KJ_PER_KCAL * _ap.aux_term_kcal(_z, _rec['aux_center'], _rec['aux_k_kcal_mol'])
            if hasattr(block, 'flush'):
                block.flush()
        if low_memory:
            u_nk_block_paths.append(block)
        else:
            all_unk_blocks.append(block)
        if memory_reporter is not None:
            memory_reporter.record(f'epoch_bias_written:{_phase_label(epoch_dir)}')
        del block, samples

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
    v_pep   = np.concatenate(all_v_pep); del all_v_pep
    v_dih   = np.concatenate(all_v_dih); del all_v_dih
    aux_z = np.concatenate(all_aux_z) if _aux_cols else None
    gamd_lambda_sample = np.concatenate(all_gamd_lambda_sample); del all_gamd_lambda_sample
    pot_arr = np.concatenate(all_potential); del all_potential
    potential = pot_arr if np.any(np.isfinite(pot_arr)) else None
    if low_memory:
        rows = sum(block.shape[0] for block in u_nk_block_paths)
        columns = u_nk_block_paths[0].shape[1]
        u_nk = create_matrix((rows, columns), directory=adaptive_dir)
        _u_nk_path = Path(u_nk.filename)
        offset = 0
        for block in u_nk_block_paths:
            for start, stop in row_blocks(*block.shape):
                u_nk[offset + start:offset + stop] = block[start:stop]
                release_pages(u_nk, written=True)
                release_pages(block)
            offset += block.shape[0]
        del block, u_nk_block_paths
        del all_unk_blocks
    else:
        u_nk = np.concatenate(all_unk_blocks, axis=0); del all_unk_blocks
    if memory_reporter is not None:
        memory_reporter.record('after_union_write')

    # state_registry.csv (state_lambdas, built above) and the per-sample
    # gamd_lambda column just concatenated here are two independent sources for
    # the same quantity. A stale/regenerated registry that reads all-zero while
    # the samples plainly carry active rungs must fail loud here, not silently
    # skip the ladder boost below and report a confident PASS.
    from .ladder import assert_lambda_sources_agree
    assert_lambda_sources_agree(state_lambdas, gamd_lambda_sample)

    epoch_src = np.concatenate(all_epoch_src) if all_epoch_src else np.zeros(len(cv), dtype=np.int32)
    del all_epoch_src

    # Drop per-state burnin frames: for each sample, compare its epoch-local step
    # against the burnin threshold for the state it was collected in.
    keep = None
    if np.any(burnin_by_k > 0):
        keep = step >= burnin_by_k[window]
        n_dropped = int((~keep).sum())
        if n_dropped > 0:
            print(f'    [burnin filter] dropped {n_dropped}/{len(cv)} samples ({100*n_dropped/len(cv):.1f}%) from pre-equilibration steps')
    _aux_burnin: dict = {}
    if _aux_cols:
        # Worker burn-in: the worker's samples of the phase(s) of its burnin_phase_epoch (not burnin_steps).
        _epochs = [_ap.phase_epoch(_aux_phase_label(Path(d), adaptive_dir)) for d in epoch_source_dirs]
        _src_epoch = np.asarray([-1 if e is None else e for e in _epochs], dtype=np.int64)
        _row_epoch = _src_epoch[epoch_src]
        _aux_keep = np.ones(len(cv), dtype=bool)
        for _k, _rec in _aux_cols.items():
            if _rec.get('burnin_phase_epoch') is None:
                continue
            _drop = (window == _k) & (_row_epoch == int(_rec['burnin_phase_epoch']))
            _aux_burnin[str(state_ids[_k])] = int(_drop.sum())
            _aux_keep &= ~_drop
        keep = _aux_keep if keep is None else (keep & _aux_keep)
    if keep is not None:
        cv = cv[keep]; cv2 = cv2[keep]; window = window[keep]
        step = step[keep]; replica = replica[keep]; boost = boost[keep]
        boost_dih = boost_dih[keep]
        v_pep = v_pep[keep]; v_dih = v_dih[keep]
        if aux_z is not None:
            aux_z = aux_z[keep]
        epoch_src = epoch_src[keep]
        pot_arr = pot_arr[keep]
        potential = pot_arr if np.any(np.isfinite(pot_arr)) else None
        u_nk = select_matrix(u_nk, keep)

    temp = 1.0 / (K_B_KJ_PER_MOL_K * beta)

    # lambda-ladder boost: fold each active state's own closed-form Pep-GaMD
    # boost into u_nk itself (see gareus.mbar_analysis.ladder) so MBAR's own
    # reweighting is exact and the downstream cumulant/exponential GaMD
    # correction (run_pmf_and_gamd_boost_report) is skipped rather than
    # double-applied on top of an already-exact u_nk. A no-op (returns u_nk
    # unchanged, sets meta['gamd_ladder']=False) on every run/registry where
    # no state carries gamd_lambda > 0.
    from .ladder import apply_ladder_boost_to_u, load_pep_gamd_envelope
    _ladder_envelope = load_pep_gamd_envelope(adaptive_dir) if np.any(state_lambdas > 0.0) else None
    _ladder_meta: dict = {}
    u_nk = apply_ladder_boost_to_u(u_nk, v_pep, v_dih, state_lambdas, _ladder_envelope, beta, _ladder_meta)
    if memory_reporter is not None:
        memory_reporter.record('after_ladder_correction')

    # Mid-campaign secondary-CV redefinition: reported, not corrected (see
    # _secondary_cv_regime_change_note).
    _regime_note = _secondary_cv_regime_change_note([ed for ed, _ in epoch_dirs], per_dir_valid_counts)
    if _regime_note:
        if _reproj is None:
            _regime_note += (' No cv2 reprojection table is present, so every column here is '
                             f'still evaluated against whichever CV2 definition was in effect '
                             f'when each row was recorded. Generate one with '
                             f'`python reproject_cv2.py {adaptive_dir}` to make every column '
                             f'evaluable in its own definition.')
        print(f'    {_regime_note}')
        load_notes = load_notes + [_regime_note]

    if _reproj_declined:
        _bits = [f"{_phase_label(Path(_ed))}: {_c['declined_regime']} coverage "
                 f"{100*_c['declined_coverage']:.1f}% (would drop {_c['would_drop']:,} of "
                 f"{_c['rows']:,} rows)" for _ed, _c in _reproj_declined.items()]
        _dec_note = ('[cv2 reprojection] DECLINED for ' + '; '.join(_bits)
                     + f". A foreign regime's cv2 must be recoverable for at least "
                       f"{100*CV2_REPROJECTION_MIN_COVERAGE:.0f}% of a phase's rows before the "
                       f"reprojection may be used: an uncovered row gets NaN in that regime's "
                       f"u_nk columns and is then DROPPED, so a sparse table does not blur the "
                       f"pooled solve -- it deletes samples. Every sample is kept instead, under "
                       f"each epoch's own native cv2, which across a regime change is the "
                       f"splice reported above. To use the reprojection, rebuild the table with "
                       f"full coverage (reproject_cv2.py --trajectories).")
        print(f'    {_dec_note}')
        load_notes = load_notes + [_dec_note]

    if _reproj is not None and not _reproj_declined:
        # Say plainly which columns became evaluable and how much of each epoch
        # the table could actually cover: a pooled solve is only as valid as the
        # rows whose foreign cv2 was recoverable, and the rest are NaN.
        _cov_bits = []
        for _ed, _cov in _reproj_coverage.items():
            _cov_bits.append(f'{_phase_label(Path(_ed))}: '
                             + ', '.join(f'{r}={100 * c:.1f}%' for r, c in sorted(_cov.items())))
        _reproj_note = ('[cv2 reprojection] Using ' + str(Path(adaptive_dir) / CV2_REPROJECTION_FILENAME)
                        + f' with regimes {sorted(_reproj)}: each u_nk column is evaluated in the '
                          'CV2 definition its own secondary params were written for, so the pooled '
                          'solve is one Hamiltonian rather than a splice. Per-phase coverage of the '
                          'FOREIGN regime(s) -- rows outside it are NaN and get dropped: '
                        + ('; '.join(_cov_bits) if _cov_bits else 'none'))
        print(f'    {_reproj_note}')
        load_notes = load_notes + [_reproj_note]

    meta_out = dict(meta)
    if load_notes:
        meta_out['load_notes'] = list(meta.get('load_notes') or []) + load_notes
    meta_out.update({'temperature_K': temp, 'beta_1_over_kJ_mol': beta,
                     'adaptive_union_states': K, 'adaptive_union_epochs': len(epoch_dirs),
                     'umbrella_window_rows': list(reg_rows),
                     '_epoch_source': epoch_src.tolist(),
                     'adaptive_epoch_run_dirs': [str(ed) for ed, _ in epoch_dirs],
                     '_epoch_source_run_dirs': list(epoch_source_dirs)})
    if low_memory and isinstance(u_nk, np.memmap):
        meta_out['_u_nk_temp_path'] = str(_u_nk_path)
    if low_memory and (analysis_stride > 1 or analysis_stride_offset > 0):
        meta_out['analysis_stride'] = int(analysis_stride)
        meta_out['analysis_stride_offset'] = int(analysis_stride_offset)
        meta_out['analysis_stride_applied_in_loader'] = True
    meta_out.update(_ladder_meta)
    if _aux_cols:
        meta_out['aux_states'] = sorted(int(k) for k in _aux_cols)
        meta_out['aux_model_sha256'] = str(_aux_rec['model_sha256'])
        # The Stage D audit flag (pmf.aux_diagnostics_unavailable): ladder crosscheck / ladder_overlap / overlap
        # grades report unavailable and the banner is capped at CAUTION (final fix wave I1).
        meta_out['aux_models'] = [str(_aux_rec['model_sha256'])]
        meta_out['aux_burnin_dropped'] = dict(_aux_burnin)
        meta_out['aux_parent_state'] = {int(k): state_id_to_k.get(int(rec['spawn_parent_state_id']))
                                        if rec.get('spawn_parent_state_id') is not None else None
                                        for k, rec in _aux_cols.items()}
        meta_out['aux_forecast_O'] = {int(k): ((rec.get('forecast') or {}).get('O')) for k, rec in _aux_cols.items()}

    _boost_dih_arg = boost_dih if np.any(np.isfinite(boost_dih)) else None
    result = clean(Data(
        prod_dir=adaptive_dir, out_dir=adaptive_dir / 'pmf_analysis',
        cv=cv, cv2=cv2, rg_A=np.full(cv.shape, np.nan),
        window=window, replica=replica, step=step,
        u_nk=u_nk, centers=primary_centers, k_kcal=primary_ks,
        beta=beta, temp=temp, boost_kj=boost, potential_kj=potential,
        source=str(registry_csv), meta=meta_out,
        boost_dih_kj=_boost_dih_arg,
        v_pep_kj=v_pep, v_dih_kj=v_dih, state_lambdas=state_lambdas, aux_z=aux_z,
    ))
    if memory_reporter is not None:
        memory_reporter.record('after_loader_clean')
    return result
