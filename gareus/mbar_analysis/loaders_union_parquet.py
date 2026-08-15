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
