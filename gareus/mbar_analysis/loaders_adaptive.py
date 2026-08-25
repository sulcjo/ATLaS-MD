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
import os
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


# ---------------------------------------------------------------------------
# Stale epoch_window_map.csv detection + in-memory repair
#
# `--us-auto-drop-bad-windows` prunes umbrella windows *post-pull*, and
# gareus/production.py then renumbers every window-indexed array around the
# survivors (0..N-1) -- but the phase's own `epoch_window_map.csv` was already
# written by the state registry as an IDENTITY map over all *active* states,
# and nothing rewrites it (the drop is recorded only in the phase's
# `gareus_metadata.json`; `active_states()` keeps handing out the dropped
# state forever). Every local window index at or after the first dropped index
# therefore resolves to the WRONG state, and the union loader used to trust the
# map's row count blindly.
#
# On the real run this was found on (RUNS/chignolin_6) that silently
# mis-attributed ~1.82M of 12.1M samples (15%): one state was scored against a
# sign-flipped CV2 centre (2,012 kT of fabricated bias), another was never
# sampled at all yet still got an f_k, and a third pooled three different
# Hamiltonians -- collapsing MBAR's base ESS to 0.74%. Full writeup, including
# the per-phase ground truth these functions are tested against:
# docs/chignolin_6_low_ess_root_cause.md.
#
# Repairing the map *in memory* (rather than only refusing to load) is the
# whole point: it lets an already-finished multi-day run be re-analysed
# correctly without re-running any MD.
# ---------------------------------------------------------------------------

_STALE_WINDOW_MAP_DOC = 'docs/chignolin_6_low_ess_root_cause.md'
_STALE_WINDOW_MAP_OVERRIDE_ENV = 'GAREUS_ALLOW_STALE_WINDOW_MAP'

# Where a repaired map parks the rows of states that this phase dropped.
#
# The repair has to satisfy two consumers of the SAME returned row list, which
# want opposite things (`_load_epoch_task`, gareus/mbar_analysis/
# loaders_union_parquet.py):
#
#   * `wmap = {int(r['epoch_window']): int(r['state_id'])}` -- the local
#     window index -> state_id lookup for THIS phase's samples. Must contain
#     only the surviving windows, compacted 0..N-1, or samples land on the
#     wrong state (the whole point of the repair).
#   * `_parse_epoch_window_map_native_params(rows)` -- keyed purely by
#     state_id, never reads `epoch_window`. It must keep seeing EVERY original
#     row, dropped states included: MBAR still evaluates a bias column for a
#     dropped state against every *other* epoch's samples, and this snapshot is
#     the only record of the window params that state really had while this
#     epoch ran. Deleting those rows silently falls that column back to the
#     final-registry row, which is exactly the stale-snapshot bug the
#     2026-08-04 per-epoch-native-params fix exists to prevent (65-125 kT of
#     spurious bias energy per sample on a recentered state -- see CLAUDE.md).
#     The flat `epoch_NNN/epoch_window_map.csv` maps written by
#     `registry.write_epoch_window_map` carry primary_k/secondary_k as well as
#     both centres and really do differ from the registry in every row
#     (chignolin_6/epoch_000: secondary_center -2.1337 in the map vs -0.9058 in
#     the registry, 20/20 rows), so this is not hypothetical.
#
# So a dropped row is kept but renumbered far out of band instead of removed.
# The base is deliberately non-negative (a negative key would push
# `_vectorized_map_lookup` off its dense-LUT fast path onto a per-element
# Python loop over millions of samples) and far beyond any real local window
# index (real runs: tens to low hundreds), so no sample can ever resolve to a
# phantom row even if the coverage check below were somehow bypassed, and any
# code that mistakes `max(wmap)`/`len(rows)` for a window count gets an
# obviously absurd number rather than a plausible wrong one. NOTE neither the
# returned row count nor the max key is a window count -- inferring one from
# the other is precisely the bug being repaired here.
#
# Cost of the out-of-band base: `_vectorized_map_lookup`'s dense LUT is sized to
# the largest key, so a repaired phase's lookup allocates ~8 MB (1e6 int64)
# instead of a few hundred bytes -- once per repaired epoch block, against
# millions of samples, and well under _VECTORIZED_LOOKUP_MAX_TABLE_SIZE.
# That last clause is load-bearing rather than incidental, and these two
# constants were chosen independently of each other: see the assertion next to
# _VECTORIZED_LOOKUP_MAX_TABLE_SIZE below, which ties them together so raising
# this base past that cap cannot silently retire the vectorized remap.
_PHANTOM_EPOCH_WINDOW_BASE = 1_000_000


def _phase_label(phase_dir) -> str:
    """Short human-readable phase name, e.g. ``final/baseline``."""
    p = Path(phase_dir)
    return f'{p.parent.name}/{p.name}' if p.parent.name else p.name


def _phase_dropped_post_pull_windows(phase_dir) -> Optional[list]:
    """The phase's ``dropped_post_pull_bad_windows`` record, or None.

    These are **pre-drop local window indices** (the indices the auto-drop saw
    before it renumbered the survivors), not state_ids -- see
    `_map_positions_to_state_ids`. `gareus/production.py` writes the list into
    both its `window_metadata` and `secondary_cv_metadata` dicts, but which of
    the two actually survives into `gareus_metadata.json` varies by phase (on
    the real chignolin_6 run, `final/baseline` has it under `window_metadata`
    and `epoch_001/baseline` only under `secondary_cv`), so both are checked.
    """
    meta = rjson(Path(phase_dir) / 'gareus_metadata.json', {})
    for section in ('window_metadata', 'secondary_cv', 'secondary_cv_metadata'):
        block = meta.get(section)
        if not isinstance(block, dict):
            continue
        dropped = block.get('dropped_post_pull_bad_windows')
        if not isinstance(dropped, list) or not dropped:
            continue
        try:
            return sorted({int(x) for x in dropped})
        except (TypeError, ValueError):
            continue
    return None


def _phase_recorded_window_count(phase_dir) -> Optional[int]:
    """How many umbrella windows this phase *physically ran*, from its own
    post-drop metadata, or None if it recorded neither.

    `analysis_metadata_validation.json`'s `n_windows` is preferred: it is
    written at the end of the run from the real post-drop window table (it
    correctly held 24 and 23 for the two chignolin_6 baselines whose maps
    claimed 27 and 24). `gareus_metadata.json`'s `windows_A` is the same
    post-drop truth from the setup side and covers phases that never got as
    far as the validation writer. NOTE `window_metadata.explicit_window_table.
    n_windows` is deliberately NOT used -- that one is itself stale on the
    affected runs (27 on a 24-window phase).
    """
    p = Path(phase_dir)
    val = rjson(p / 'analysis_metadata_validation.json', {}).get('n_windows')
    try:
        n = int(val)
        if n > 0:
            return n
    except (TypeError, ValueError):
        pass
    windows_a = rjson(p / 'gareus_metadata.json', {}).get('windows_A')
    if isinstance(windows_a, list) and windows_a:
        return len(windows_a)
    return None


def _read_epoch_window_map_rows(phase_dir) -> Optional[list]:
    """Parse one phase's own ``epoch_window_map.csv``, or None if absent."""
    path = Path(phase_dir) / 'epoch_window_map.csv'
    if not path.exists():
        return None
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def _row_state_id(row: dict) -> Optional[int]:
    try:
        return int(row['state_id'])
    except (KeyError, TypeError, ValueError):
        return None


def _sorted_map_rows(rows: list) -> list:
    """Map rows in ascending ``epoch_window`` order (file order as fallback).

    The positional interpretation of the drop record depends on this order, so
    it is made explicit rather than inherited from however the CSV happens to
    be written.
    """
    def _key(item):
        i, r = item
        try:
            return (0, int(r['epoch_window']), i)
        except (KeyError, TypeError, ValueError):
            return (1, 0, i)
    return [r for _i, r in sorted(enumerate(rows), key=_key)]


def _map_positions_to_state_ids(map_rows: list, positions) -> Optional[set]:
    """Translate pre-drop local window indices into state_ids via the map they
    were recorded against.

    This indirection is load-bearing, not decoration: the drop record holds
    *positions*, and position == state_id only for an identity map. The
    registry's `active_states()` skips retired states, so real maps are often
    non-identity (chignolin_6's topup maps look like ``11->12, 12->15``) --
    reading the record as "state_id in dropped" instead would silently delete
    the wrong rows on any run that ever retired a state.
    """
    ordered = _sorted_map_rows(map_rows)
    out = set()
    for pos in positions:
        if pos < 0 or pos >= len(ordered):
            return None
        try:
            out.add(int(ordered[pos]['state_id']))
        except (KeyError, TypeError, ValueError):
            return None
    return out or None


def _dropped_state_ids_for_phase(phase_dir) -> Optional[tuple]:
    """``(dropped_state_ids, record_source_dir)`` for one phase, or None.

    A phase's own record is preferred. A baseline/topup_* sub-run that has no
    record of its own inherits the one from a sibling sub-run of the same
    parent phase (baseline first, then the topups): the auto-drop runs in
    whichever sub-run did the pulling, and the sibling's own map is what
    translates its positional record into state_ids. Callers MUST still verify
    that removing those state_ids reconciles the map's row count with the
    phase's real window count -- an inherited record is an inference, not a
    fact (a later topup can legitimately re-pull a window an earlier sub-run
    dropped; chignolin_6's `final/topup_001` really does sample state 23,
    which `final/baseline` dropped).
    """
    p = Path(phase_dir)
    own = _phase_dropped_post_pull_windows(p)
    if own:
        rows = _read_epoch_window_map_rows(p)
        ids = _map_positions_to_state_ids(rows, own) if rows else None
        if ids:
            return ids, p
    parent = p.parent
    siblings = []
    baseline = parent / 'baseline'
    if baseline.is_dir() and baseline != p:
        siblings.append(baseline)
    siblings += [d for d in sorted(parent.glob('topup_*')) if d.is_dir() and d != p]
    for sib in siblings:
        dropped = _phase_dropped_post_pull_windows(sib)
        if not dropped:
            continue
        rows = _read_epoch_window_map_rows(sib)
        ids = _map_positions_to_state_ids(rows, dropped) if rows else None
        if ids:
            return ids, sib
    return None


def _row_float(row: dict, *keys) -> float:
    """First parseable float among `keys` on `row`, else NaN."""
    for k in keys:
        v = row.get(k, '')
        if v in ('', 'None', 'nan', None):
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return float('nan')


def _phase_real_window_centers(phase_dir) -> Optional[list]:
    """``[(primary_center, secondary_center)]`` per *real* local window, from the
    phase's own ``umbrella_explicit_windows.csv``, or None if it has none.

    This is the window table the sampler actually ran: it is rewritten after
    the post-pull auto-drop (`gareus/production.py` calls
    `write_explicit_window_analysis_files` with the surviving windows), so its
    rows are the true post-drop local windows 0..N-1 with the centres each one
    was really restrained at. It is therefore a *direct* record of what ran,
    strictly better evidence than the drop record (which only says which
    pre-drop indices went away, and which several real runs on disk either
    never wrote or wrote incompletely).
    """
    path = Path(phase_dir) / 'umbrella_explicit_windows.csv'
    if not path.exists():
        return None
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    return [(_row_float(r, 'primary_center', 'distance_center_A'),
             _row_float(r, 'secondary_cv_center', 'secondary_center')) for r in rows]


def _match_map_rows_to_real_windows(map_rows: list, real_centers: list,
                                    tol: float = 1e-6) -> Optional[list]:
    """The subsequence of `map_rows` whose centres are the real windows', in
    order -- i.e. the map with its phantom rows removed. None if any real
    window has no match left, which means these two artifacts do not describe
    the same phase and nothing may be inferred from them.

    Order-preserving on purpose: the auto-drop renumbers the survivors without
    reordering them, so the correct answer is always a subsequence, and
    matching greedily left-to-right can never "reach back" past a window it
    already consumed. Centres come from the same in-memory floats written to
    both CSVs, so the tolerance only has to absorb text round-tripping -- it is
    orders of magnitude below any real window spacing (the tightest CV2 gap on
    the runs this was built against is ~0.02).
    """
    def _close(a: float, b: float) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol + tol * max(abs(a), abs(b))

    picked = []
    j = 0
    for c1, c2 in real_centers:
        while j < len(map_rows):
            r = map_rows[j]
            j += 1
            if (_close(_row_float(r, 'primary_center'), c1)
                    and _close(_row_float(r, 'secondary_center'), c2)):
                picked.append(r)
                break
        else:
            return None
    return picked


def _renumber_epoch_window_rows(rows: list, start: int = 0) -> list:
    """Copy of `rows` with ``epoch_window`` renumbered ``start..start+N-1`` in order.

    With the default `start=0` this is exactly what the writer would have
    produced had it enumerated the phase's surviving windows instead of the
    registry's active states. Everything else on the row (state_id,
    primary/secondary centre, any k columns) is carried through untouched, and
    the inputs are never mutated. `start` is used to park the *dropped* rows
    out of band -- see `_PHANTOM_EPOCH_WINDOW_BASE`.
    """
    out = []
    for i, r in enumerate(_sorted_map_rows(rows)):
        new = dict(r)
        new['epoch_window'] = str(start + i)
        out.append(new)
    return out


def _per_window_cv2_means(window_ids, cv2) -> dict:
    """``{local_window_index: mean sampled cv2}``, skipping non-finite values."""
    if window_ids is None or cv2 is None:
        return {}
    w = np.asarray(window_ids).astype(np.int64, copy=False).ravel()
    c = np.asarray(cv2, dtype=np.float64).ravel()
    if w.size == 0 or w.size != c.size:
        return {}
    finite = np.isfinite(c)
    if not np.any(finite):
        return {}
    w = w[finite]; c = c[finite]
    order = np.argsort(w, kind='stable')
    w = w[order]; c = c[order]
    uniq, starts = np.unique(w, return_index=True)
    sums = np.add.reduceat(c, starts)
    counts = np.diff(np.append(starts, c.size))
    return {int(k): float(s / n) for k, s, n in zip(uniq, sums, counts) if n > 0}


def _map_cv2_fingerprint_deviation(map_rows: list, cv2_means: dict) -> Optional[float]:
    """Mean ``|sampled cv2 - assigned secondary_center|`` over local windows.

    The independent, physics-side evidence that a local-window -> state mapping
    is right: a harmonically restrained window's samples sit on their own
    restraint centre, so this is ~sigma for a correct mapping and grossly
    larger for a shifted one (the chignolin_6 shift put a window's samples on
    the *opposite sign* of their assigned CV2 centre). Returns None when there
    is nothing to compare -- a CV1-only run has no sampled cv2 at all, and a
    map row may carry no secondary centre.
    """
    if not cv2_means:
        return None
    devs = []
    for r in map_rows:
        try:
            ew = int(r['epoch_window'])
            center2 = float(r.get('secondary_center', ''))
        except (KeyError, TypeError, ValueError):
            continue
        if ew not in cv2_means or not math.isfinite(center2):
            continue
        devs.append(abs(cv2_means[ew] - center2))
    return float(np.mean(devs)) if devs else None


def _validate_and_repair_epoch_window_map(phase_dir, wmap_rows: list,
                                          window_ids=None, cv2=None) -> tuple:
    """``(rows, notes)`` for one adaptive-production phase's window map.

    Returns `wmap_rows` unchanged (and no notes) when the map's row count
    agrees with how many windows the phase really ran -- the overwhelming
    majority of phases, including every phase of every run that never used
    `--us-auto-drop-bad-windows`.

    On a mismatch the map is stale (see this section's header comment) and the
    phantom rows are identified from whichever of two independent records the
    phase left behind -- preferring the direct one:

    1. the phase's own surviving window table (``umbrella_explicit_windows.csv``),
       matched onto the map by restraint centre -- this is what actually ran, and
       it is the only source that covers a phase whose drop record is missing or
       does not account for the whole gap (both occur on real runs on disk);
    2. failing that, the post-pull ``dropped_post_pull_bad_windows`` record --
       the phase's own, or an inherited sibling sub-run's -- applied by state_id
       and accepted only if the surviving row count reconciles.

    A successful repair rebuilds the map the way the writer should have written
    it (survivors renumbered 0..N-1, original state_ids and centres kept) and
    returns a loud note. The dropped states' rows are NOT deleted: they are
    kept, verbatim except for an out-of-band ``epoch_window`` no sample can
    reach, so that state-keyed consumers of the same list still see this
    phase's native window params for them -- see `_PHANTOM_EPOCH_WINDOW_BASE`
    for why that matters. Consequently `len(returned_rows)` is NOT this phase's
    window count (it never was a safe inference; assuming it was is the bug
    being repaired). If neither record yields a reconciling repair, or the
    repair makes the sampled-CV2 fingerprint *worse*, this raises ValueError:
    silently attributing samples to the wrong umbrella state is far worse than
    refusing to load. ``GAREUS_ALLOW_STALE_WINDOW_MAP=1`` downgrades that
    refusal to a warning and loads with the stale map anyway (escape hatch for
    inspecting a damaged run; the free energies are then wrong by construction).
    It never suppresses a repair that *is* possible.

    `window_ids`/`cv2` are this phase's own sample columns, used for the
    metadata-free row-count check and the CV2 fingerprint. Both optional.
    """
    label = _phase_label(phase_dir)
    rows = _sorted_map_rows(wmap_rows)
    n_map = len(rows)
    cv2_means = _per_window_cv2_means(window_ids, cv2)

    observed = None
    if window_ids is not None:
        w = np.asarray(window_ids)
        if w.size:
            observed = int(w.astype(np.int64, copy=False).max()) + 1
    recorded = _phase_recorded_window_count(phase_dir)

    # Prefer the phase's own post-drop metadata as the real window count: it
    # counts windows that *ran*, whereas the sample-derived count undercounts a
    # window that happened to produce no samples (interrupted phase, dead
    # replica). Fall back to the samples when there is no metadata at all --
    # that check needs nothing but the Parquet data itself.
    n_true = recorded if recorded is not None else observed
    if n_true is None:
        return list(wmap_rows), []
    count_source = ("the phase's own recorded window count"
                    if recorded is not None else 'the distinct window_id values in its Parquet samples')

    if n_map == n_true and (observed is None or observed <= n_map):
        return list(wmap_rows), []

    detail = (f'epoch_window_map.csv for adaptive-production phase {label} lists {n_map} '
              f'window(s) but the phase physically ran {n_true} ({count_source})')
    if observed is not None and recorded is not None and observed != recorded:
        detail += f'; its samples reference {observed} distinct local window index(es)'

    def _fail(reason: str):
        msg = (f'{detail}. {reason} Loading anyway would attribute this phase\'s samples to the '
               f'wrong umbrella states (the auto-drop renumbering bug written up in '
               f'{_STALE_WINDOW_MAP_DOC}), so the free energies would be silently wrong -- '
               f'refusing to load. Set {_STALE_WINDOW_MAP_OVERRIDE_ENV}=1 to load with the '
               f'stale map regardless (for inspection only).')
        if os.environ.get(_STALE_WINDOW_MAP_OVERRIDE_ENV, '').strip().lower() in ('1', 'true', 'yes'):
            return list(wmap_rows), [
                f'[stale window map] {detail}. {reason} Loaded with the STALE map anyway because '
                f'{_STALE_WINDOW_MAP_OVERRIDE_ENV} is set: this phase\'s samples are attributed to '
                f'the wrong umbrella states and every free energy derived from them is invalid. '
                f'See {_STALE_WINDOW_MAP_DOC}.']
        raise ValueError(msg)

    # Preferred repair source: the phase's own surviving window table, matched
    # onto the map by centre (direct evidence of what ran -- see
    # _phase_real_window_centers). Only trusted when its length agrees with the
    # phase's real window count, otherwise it is itself stale/partial.
    #
    # `kept_rows` is the subsequence of `rows` belonging to windows that really
    # ran; `phantom_src_rows` is everything else. Both halves are needed: only
    # the first may appear in the local-index lookup, and both must survive
    # into the returned list so the state-keyed native-params parser still sees
    # the dropped states' epoch-native window params (see
    # _PHANTOM_EPOCH_WINDOW_BASE -- deleting them re-creates the stale-registry
    # bias bug for those states' columns).
    kept_rows = None
    repair_src = ''
    real_centers = _phase_real_window_centers(phase_dir)
    if real_centers is not None and len(real_centers) == n_true:
        matched = _match_map_rows_to_real_windows(rows, real_centers)
        if matched is not None:
            kept_rows = matched
            repair_src = 'its own surviving window table (umbrella_explicit_windows.csv)'

    # Fallback: the post-pull drop record (this phase's own, or an inherited
    # sibling's), applied by state_id and accepted only if the surviving row
    # count reconciles with the real window count.
    if kept_rows is None:
        found = _dropped_state_ids_for_phase(phase_dir)
        if found is None:
            return _fail('Neither a usable umbrella_explicit_windows.csv nor a '
                         'dropped_post_pull_bad_windows record was found for it (the latter neither '
                         'in its own gareus_metadata.json nor in a sibling sub-run of the same '
                         'phase), so the correct local-window -> state_id mapping cannot be '
                         'reconstructed.')
        dropped_ids, record_src = found
        kept = [r for r in rows if _row_state_id(r) not in dropped_ids]
        if len(kept) != n_true:
            return _fail(f'Its drop record (state_id(s) {sorted(dropped_ids)}, from '
                         f'{_phase_label(record_src)}) removes {n_map - len(kept)} of the {n_map} '
                         f'rows, leaving {len(kept)} -- which does not reconcile with {n_true}, so '
                         'the record does not describe this map and must not be applied.')
        kept_rows = kept
        repair_src = (f'the drop record in {_phase_label(record_src)} '
                      f'(dropped state_id(s) {sorted(dropped_ids)})')

    # Identity, not equality: two windows of the same phase can legitimately
    # share a centre pair (a CV1-only ladder repeats secondary_center), so the
    # complement must be taken over the actual row objects the matcher picked.
    kept_ids = {id(r) for r in kept_rows}
    phantom_src_rows = [r for r in rows if id(r) not in kept_ids]
    survivors = _renumber_epoch_window_rows(kept_rows)
    phantoms = _renumber_epoch_window_rows(phantom_src_rows, start=_PHANTOM_EPOCH_WINDOW_BASE)

    # Post-repair coverage: a sample pointing past the *surviving* windows
    # would be remapped to the -1 sentinel and silently dropped by the caller
    # (it can never reach a phantom row), which would mean the "authoritative"
    # window count is itself wrong. Counted over `survivors` alone on purpose:
    # len(survivors) + len(phantoms) is still the stale row count.
    if observed is not None and observed > len(survivors):
        return _fail(f'Its samples reference local window index {observed - 1}, beyond the '
                     f'{len(survivors)} window(s) the repaired map would cover.')

    # Independent physics-side check on the repair (see
    # _map_cv2_fingerprint_deviation): a correct mapping puts each window's
    # samples on their own restraint centre. Only the survivors carry real
    # local indices, so only they can be compared against sampled cv2.
    dev_before = _map_cv2_fingerprint_deviation(rows, cv2_means)
    dev_after = _map_cv2_fingerprint_deviation(survivors, cv2_means)
    if dev_before is not None and dev_after is not None and dev_after > dev_before:
        return _fail(f'The repair implied by {repair_src} would move the sampled cv2 values further '
                     f'from their assigned restraint centres (mean |cv2 - secondary_center| '
                     f'{dev_before:.4f} -> {dev_after:.4f}), i.e. it does not describe '
                     "this phase's window set.")
    if dev_before is not None and dev_after is not None:
        evidence = (f' Sampled-cv2 fingerprint confirms it: mean |cv2 - secondary_center| '
                    f'{dev_before:.4f} -> {dev_after:.4f}.')
    else:
        evidence = (' No sampled cv2 was available to cross-check the repair against '
                    '(CV1-only phase, or no secondary centres in the map).')

    shifted = [f'{i}->{_row_state_id(r)}' for i, r in enumerate(survivors)
               if _row_state_id(r) != _row_state_id(rows[i])]
    # Whether anything was actually mis-attributed depends on *where* the
    # phantom rows sat: only rows before a sampled local index shift the ones
    # after them, so a phase whose phantom rows all sit past its last real
    # window (seen on real runs) had its own samples on the right states
    # already. Say which case this is rather than implying damage either way --
    # but do not overclaim in the second case: this function only ever sees one
    # phase, while MBAR solves the whole campaign as one joint f_k problem, so
    # an unshifted phase says nothing about whether an earlier *run-level*
    # result was valid.
    if shifted:
        impact = (f'corrected local->state mapping for the {len(shifted)} shifted window(s): '
                  f'{", ".join(shifted)}. Every sample in those windows was previously '
                  f'attributed to the wrong umbrella state, so any earlier analysis of this '
                  f'phase is invalid.')
    else:
        impact = ('no local window changed state: every phantom row sat past this phase\'s last '
                  'real window, so the repair does not move any of this phase\'s own samples. '
                  'That alone does NOT validate an earlier analysis of the run: MBAR solves every '
                  'phase jointly, so a shifted mapping in any other phase -- or a never-sampled '
                  'phantom state entering the solve at all -- still invalidates the result.')
    phantom_note = ''
    if phantoms:
        phantom_note = (f' The {len(phantoms)} dropped state(s) '
                        f'{sorted(sid for sid in (_row_state_id(r) for r in phantoms) if sid is not None)} '
                        f'are kept out of the local-window lookup but retained as rows, so their '
                        f'epoch-native window params still back their MBAR bias columns instead of '
                        f'falling back to the (possibly recentered) final registry.')
    note = (f'[stale window map] {detail} -- `--us-auto-drop-bad-windows` dropped windows post-pull '
            f'and renumbered the survivors 0..{n_true - 1}, but this map was never rewritten. '
            f'REPAIRED IN MEMORY from {repair_src}; {impact}'
            f'{evidence}{phantom_note} The run data on disk is unchanged and still stale; see '
            f'{_STALE_WINDOW_MAP_DOC}.')
    return survivors + phantoms, [note]


# A lookup table is only built when the key range actually needed (mapping
# keys unioned with the array's own value range) stays small -- otherwise a
# sparse key space (e.g. one huge outlier ID) would turn a memory-savings
# fix into a memory blowup. Above this, fall back to the original per-element
# Python-level lookup, which stays correct (just not vectorized) regardless
# of key sparsity.
_VECTORIZED_LOOKUP_MAX_TABLE_SIZE = 10_000_000

# The one place the two independently chosen constants of this module actually
# meet, which nothing else expresses.
#
# A repaired window map hands the caller ONE row list, phantoms included, and
# the caller (`_load_epoch_task`, gareus/mbar_analysis/loaders_union_parquet.py)
# turns all of it into the `wmap` dict that `_vectorized_map_lookup` then
# applies to that phase's whole window_id column (millions of rows per phase;
# 12,103,761 samples campaign-wide on chignolin_6). So it is the parked phantom
# keys, not the two dozen real ones, that size the LUT: it spans
# `max(key) + 1 == _PHANTOM_EPOCH_WINDOW_BASE + n_phantoms`. Staying under the
# cap is what keeps the vectorized path in play; cross it and that
# multi-million-row remap silently drops to the per-element Python loop. Note
# this is NOT the sparse-key blowup the comment above warns about -- the guard
# is doing its job here. The risk is only that a future tuning of either number
# quietly gives up the fast path with nothing complaining.
#
# `base < cap` rather than `base + n_phantoms <= cap`: n_phantoms is how many
# windows a single phase dropped (three on the run this was built for, tens at
# the very worst), and the ~9M of headroom the current values leave makes that
# count irrelevant. Anything that tightens the base or loosens the cap far
# enough for the phantom count to matter trips this check first, so leave the
# headroom rather than shaving it.
#
# The alternative to coupling the constants is to stop the phantom keys reaching
# the lookup dict at all -- i.e. have `_load_epoch_task` skip rows whose
# `epoch_window >= _PHANTOM_EPOCH_WINDOW_BASE` when it builds `wmap`, which
# would drop the LUT back to a couple of hundred bytes. That is safe with
# respect to the per-epoch-native-params fix (`native_params` is built from the
# full row list on the following line and is keyed purely by state_id, never
# reading `epoch_window`), so it is a legitimate improvement -- but it belongs
# in the caller, and it would NOT retire this check: the contract of the
# returned rows still parks phantoms out of band, so any future consumer that
# builds a dict over all of them re-creates the same LUT sizing question.
#
# This assertion is not independently revert-checkable -- deleting it changes no
# behaviour on its own. The executable guard is
# tests/test_stale_epoch_window_map_guard.py::
# test_repaired_map_keeps_the_vectorized_lookup_fast_path, which measurably
# fails (rather than merely restating the inequality) once the two drift far
# enough apart to abandon the LUT.
assert _PHANTOM_EPOCH_WINDOW_BASE < _VECTORIZED_LOOKUP_MAX_TABLE_SIZE, (
    f'_PHANTOM_EPOCH_WINDOW_BASE ({_PHANTOM_EPOCH_WINDOW_BASE}) must stay below '
    f'_VECTORIZED_LOOKUP_MAX_TABLE_SIZE ({_VECTORIZED_LOOKUP_MAX_TABLE_SIZE}): a '
    'repaired epoch window map parks its dropped states\' rows at the phantom base, '
    'and the caller feeds those keys into _vectorized_map_lookup over the phase\'s '
    'whole window_id column -- at or above the cap, that remap falls back to a '
    'per-element Python loop over millions of samples.')


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
