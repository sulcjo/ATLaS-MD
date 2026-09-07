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

# What a note from this guard actually REPORTS, as a value rather than as a
# phrase a consumer has to recognise in the text.
#
# `_validate_and_repair_epoch_window_map` returns `(rows, notes)`, and for a
# while "notes is non-empty" meant exactly one thing: the map was stale. It no
# longer does -- the equal-row-count branch can now report that a *check could
# not run* on a phase whose map is not known to be wrong at all. Two consumers
# in gareus/mbar_analysis/loaders.py had keyed off the old meaning, so the
# weakest thing this guard can say ("I could not cross-check this") was
# reaching `check_union_npz_window_map_provenance`'s stale bucket and raising
# -- refusing to load a run on the strength of an absent check, which inverts
# the severity the note itself is graded at (gareus_report.py deliberately
# triages the "Could not cross-check" wording one band BELOW a detected fault).
#
# Hence these kinds. They are carried on the note object itself, so a consumer
# branches on `window_map_note_reports_a_fault(note)` rather than on a regex
# over prose that is rewritten every time the diagnosis improves -- prose
# coupling is how the drift happened in the first place. `WindowMapNote` is a
# `str` subclass, so every existing consumer (printing, `meta['load_notes']`,
# JSON-serialising into pmf_summary.json's warnings, gareus_report.py's own
# regex triage of that JSON) keeps working untouched and unaware.
#
# The kind does NOT survive `str()`/f-string interpolation of a note, which is
# why every branch below must be taken on the note object itself, before any
# copy of its text is made. A note that reaches a consumer without a kind
# (a plain str built elsewhere, a copy) grades as MAP_NOTE_UNKNOWN, which is
# treated as a fault: an unlabelled note falls back to the old, stricter
# meaning rather than to the permissive one.
MAP_NOTE_REPAIRED = 'repaired'                 # rows rewritten; the map on disk is wrong
MAP_NOTE_STALE_LOADED = 'stale_loaded'         # map is wrong and was loaded anyway (override)
MAP_NOTE_FAULT_UNREPAIRED = 'fault_unrepaired'  # fault detected, rows unchanged, not derivable
MAP_NOTE_UNCHECKED = 'unchecked'               # a check could not run; nothing detected
MAP_NOTE_UNKNOWN = 'unknown'                   # unlabelled note; treated as a fault


class WindowMapNote(str):
    """A window-map note that also says, structurally, what kind of thing it is.

    Behaves as its own text everywhere (see the kinds above for why that
    matters); `note.kind` is the machine-readable half. Deliberately not a
    dataclass or a tuple: the notes travel through half a dozen consumers that
    print them, extend lists with them and JSON-serialise them, and none of
    those may need changing to add a distinction only two of them care about.
    """

    # `kind` MUST keep a default, and the default must be the fail-closed one.
    # `copy`, `copy.deepcopy` and `pickle` all rebuild a str subclass through
    # `str.__reduce_ex__`, which calls `cls.__new__(cls, text)` -- the text
    # alone, from str's own `__getnewargs__` -- and only then restores the
    # instance `__dict__` (where `kind` really travels). With `kind` required
    # that call raises `TypeError`, so a note saying merely "I could not check
    # this phase" would abort any caller that copies the metadata dict it rides
    # in: a guard whose warnings can crash their carrier is worse than the drift
    # it was added to fix. Pinned by
    # tests/test_stale_map_other_consumers.py::
    # test_a_window_map_note_survives_being_copied_and_pickled.
    def __new__(cls, text: str, kind: str = MAP_NOTE_UNKNOWN):
        obj = super().__new__(cls, text)
        obj.kind = kind
        return obj


def window_map_note_kind(note) -> str:
    """The note's kind, or `MAP_NOTE_UNKNOWN` for a note that carries none."""
    return getattr(note, 'kind', MAP_NOTE_UNKNOWN)


def window_map_note_reports_a_fault(note) -> bool:
    """True when the note says this phase's map does NOT describe the mapping
    the phase really used -- repaired, loaded-stale-anyway, or detected and not
    repairable.

    False only for `MAP_NOTE_UNCHECKED`: a check that could not run is not a
    detected fault, and a consumer that treats it as one refuses healthy runs.
    Anything unlabelled counts as a fault, so the failure mode of a note that
    loses its kind is over-strictness, never a silently accepted bad mapping.
    """
    return window_map_note_kind(note) != MAP_NOTE_UNCHECKED


def window_map_note_rewrote_rows(note) -> bool:
    """True when the guard actually rebuilt the row list that came back with it.

    The distinction a consumer needs before it does anything positional with
    the returned rows: only after a repair are they renumbered 0..N-1 with the
    dropped states' rows parked at `_PHANTOM_EPOCH_WINDOW_BASE`.
    """
    return window_map_note_kind(note) == MAP_NOTE_REPAIRED

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


def _row_int(row: dict, key: str) -> Optional[int]:
    """`row[key]` as an int, or None when it is missing or unparseable."""
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _read_phase_window_table(phase_dir) -> tuple:
    """``(centers, problem)`` from the phase's own ``umbrella_explicit_windows.csv``.

    `centers` is ``[(primary_center, secondary_center)]`` per *real* local
    window; `problem` is a phrase describing why the table cannot be used
    positionally. Exactly one of the two is ever set, and the three outcomes
    are deliberately distinct:

    * ``(None, None)`` -- the phase has no such table at all. Not a check that
      failed: a plain 1D ladder or a legacy layout never wrote one.
    * ``(None, "...")`` -- the table exists but cannot be lined up by position.
    * ``([...], None)`` -- usable.

    This is the window table the sampler actually ran: it is rewritten after
    the post-pull auto-drop (`gareus/production.py` calls
    `write_explicit_window_analysis_files` with the surviving windows), so its
    rows are the true post-drop local windows 0..N-1 with the centres each one
    was really restrained at. It is therefore a *direct* record of what ran,
    strictly better evidence than the drop record (which only says which
    pre-drop indices went away, and which several real runs on disk either
    never wrote or wrote incompletely).

    Everything that consumes these centres consumes them POSITIONALLY -- row
    *i* is local window *i* -- so the table's own ``window`` column is checked
    to be numbered 0..N-1 in file order rather than discarded, which is the
    same guard the map's ``epoch_window`` column already gets on the other side
    of the same comparison. `explicit_window_analysis_rows`
    (`gareus/production.py`) enumerates the surviving windows in order and
    writes ``window: int(i)``, so this holds for every table any current writer
    produces and for all 291 tables under RUNS/ (audited: every one has the
    column, every one is 0..N-1 in file order). A writer that ever sorted the
    table differently while keeping the column would otherwise have every
    consumer here compare the wrong pairs, silently, on every phase. Sorting by
    the column instead of refusing is deliberately NOT done: which of the two
    orders is the local-window order would then be an inference, and this guard
    abstains rather than guesses. A table with no ``window`` column at all
    (nothing contradicts file order) is still trusted, as it always was.
    """
    path = Path(phase_dir) / 'umbrella_explicit_windows.csv'
    if not path.exists():
        return None, None
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None, None
    if 'window' in rows[0]:
        numbering = [_row_int(r, 'window') for r in rows]
        if numbering != list(range(len(rows))):
            return None, (f"the table's own `window` column is not numbered 0..{len(rows) - 1} "
                          f"in file order ({numbering[:8]}...), so its rows are not this phase's "
                          f"local windows 0..N-1 by position")
    return [(_row_float(r, 'primary_center', 'distance_center_A'),
             _row_float(r, 'secondary_cv_center', 'secondary_center')) for r in rows], None


def _phase_real_window_centers(phase_dir) -> Optional[list]:
    """The phase's real per-window centres, or None when there is no usable table.

    The repair path's view of `_read_phase_window_table`: it has no way to
    report a problem, and must not match against a table it cannot line up by
    position, so an unusable table and an absent one both come back None (the
    repair then falls through to the drop record, and failing closed from
    there is that path's documented policy). The membership check calls
    `_read_phase_window_table` directly, because for it the difference between
    the two is the difference between silence and a note.
    """
    return _read_phase_window_table(phase_dir)[0]


# How close two restraint centres must be to be the same centre. Both sides are
# the same in-memory float written into two CSVs by the same process, so this
# only has to absorb text round-tripping -- it stays orders of magnitude below
# the closest two real windows of one phase ever come. Measured over the exact
# tables this check reads (`_find_adaptive_epoch_dirs` + `_read_phase_window_table`,
# 126 sample-holding phases under RUNS/): the smallest nonzero max-axis
# separation between two windows of a phase is 4.21e-4
# (chignolin_quicktest/final/baseline, windows 1-2) and the smallest CV2-only
# gap is 6.21e-5 (chignolin_5/epoch_001/baseline) -- 420x and 62x this
# tolerance. An earlier revision of this comment quoted ~0.02 and 0.006, which
# were 14x and 320x too loose; the margin is ample either way, and the
# tolerance is unchanged. Both ratios above are against the BARE constant; the
# comparison `_centers_close` performs is relative (`tol + tol*max(|a|, |b|)`),
# under which the same two separations are 363x and 43.5x -- see
# `_permute_map_rows_onto_window_table`, whose exactness premise is stated in
# that form because that is the number its correctness actually rests on.
_CENTER_MATCH_TOL = 1e-6


def _centers_close(a: float, b: float, tol: float = _CENTER_MATCH_TOL) -> bool:
    """True when two restraint centres are the same number, to within text
    round-tripping (see `_CENTER_MATCH_TOL`).

    NaN equals NaN here on purpose: a CV1-only phase has no secondary centre on
    either side, and "absent in both records" is agreement, not disagreement.
    Hoisted to module level so the repair matcher below and the membership check
    that shares its evidence can never drift apart on what "same centre" means.
    """
    if math.isnan(a) and math.isnan(b):
        return True
    return abs(a - b) <= tol + tol * max(abs(a), abs(b))


def _match_map_rows_to_real_windows(map_rows: list, real_centers: list,
                                    tol: float = _CENTER_MATCH_TOL) -> Optional[list]:
    """The subsequence of `map_rows` whose centres are the real windows', in
    order -- i.e. the map with its phantom rows removed. None if any real
    window has no match left, which means these two artifacts do not describe
    the same phase and nothing may be inferred from them.

    Order-preserving on purpose: the auto-drop renumbers the survivors without
    reordering them, so the correct answer is always a subsequence, and
    matching greedily left-to-right can never "reach back" past a window it
    already consumed.

    GREEDY, therefore not a verdict on its own. It takes the first row whose
    centres match, so on a map LONGER than the window set it will happily
    consume a phantom row -- one for a state the phase dropped and never sampled
    -- that duplicates a real window's centres and sits earlier in the map, and
    report success. `_validate_and_repair_epoch_window_map` consequently no
    longer asks this function whether a longer map is derivable: it asks
    `_select_map_rows_onto_window_table`, which refuses ambiguity, and uses this
    one only to answer the separate, purely descriptive question "and is that
    forced selection sitting in the map's own row order?". Whoever restores this
    as a decider restores the bug.

    Still a decider where the ambiguity cannot mean a phantom: the equal-count
    membership check (`_consistent_map_membership_notes`) has as many rows as
    windows, so every candidate row is a real window of the phase -- see
    `_select_map_rows_onto_window_table`'s two bullets for why that changes the
    answer rather than merely the odds.
    """
    def _close(a: float, b: float) -> bool:
        return _centers_close(a, b, tol)

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


def _permute_map_rows_onto_window_table(map_rows: list, real_centers: list,
                                        tol: float = _CENTER_MATCH_TOL) -> Optional[list]:
    """`map_rows` REORDERED so that row *i* carries local window *i*'s real
    restraint centres, or None when the two artifacts do not list the same
    window set.

    The sibling of `_match_map_rows_to_real_windows`, for the other defect. That
    one is order-preserving because the auto-drop renumbers survivors without
    reordering them, so its answer is a subsequence. This one exists for the
    case where every window that really ran *does* have a row and only the
    order is wrong -- a permutation, which a subsequence match rejects outright
    (it cannot reach back past a row it already consumed).

    Returning a list here is what makes the equal-count membership defect
    repairable at all: local window *i*'s samples were generated at the table's
    row *i* centres, so the state_id that belongs to them is the one on the map
    row carrying those centres, whatever position that row is sitting at.

    Greedy, first-unused-match, which is exact whenever no two windows of one
    phase are within `tol` of each other on BOTH axes -- the predicate this
    matcher actually asks, and the one the margin below is measured against.

    Re-measured over all 291 usable window tables under RUNS/ (the set
    `_read_phase_window_table` accepts), against the EFFECTIVE relative
    tolerance `_centers_close` applies -- `tol + tol*max(|a|, |b|)`, not the
    bare constant: zero confusable pairs, and the closest two windows of one
    phase ever come to being confusable is 363x that effective tolerance
    (chignolin_quicktest/final/baseline windows 1-2: identical primary centre,
    4.21e-4 apart on secondary against an effective 1.16e-6). That 363x -- 2.6
    orders of magnitude, not four -- is the real margin under this premise. The
    tightest SINGLE-axis margin over the same tables is much smaller, 43.5x
    (chignolin_5/epoch_001/baseline windows 13 and 20, 6.21e-5 apart on
    secondary against an effective 1.43e-6), but that pair is separated by
    95,000x on its primary axis and so is nowhere near confusable here; a
    single-axis number is simply not what this premise rests on. An earlier
    revision of this docstring claimed "four orders of magnitude on every real
    phase" and pointed at `_CENTER_MATCH_TOL`, whose own comment quotes 420x
    and 62x against the bare constant. The substance held; the number did not.

    Two rows with *identical* centres are genuinely interchangeable here and
    either choice yields the same restraint columns; see the note text for that
    limit -- and see `_select_map_rows_onto_window_table`, which refuses that
    ambiguity outright instead of resolving it greedily, for why the two
    matchers answer it differently.
    """
    unused = list(range(len(map_rows)))
    picked = []
    for c1, c2 in real_centers:
        for pos, j in enumerate(unused):
            r = map_rows[j]
            if (_centers_close(_row_float(r, 'primary_center'), c1, tol)
                    and _centers_close(_row_float(r, 'secondary_center'), c2, tol)):
                picked.append(r)
                unused.pop(pos)
                break
        else:
            return None
    return picked if not unused else None


def _select_map_rows_onto_window_table(map_rows: list, real_centers: list,
                                       tol: float = _CENTER_MATCH_TOL) -> tuple:
    """``(picked, problem)`` -- one map row per real window, IN THE TABLE'S
    ORDER, when that assignment is FORCED by the two artifacts; ``(None,
    phrase)`` when it is not.

    The third matcher, and the single gate on EVERY map longer than the window
    set -- in the map's own row order or not. It was introduced for the compound
    defect neither sibling above covers (a map that is BOTH longer than the
    window set AND lists it out of order): `_match_map_rows_to_real_windows`
    cannot see that one (order-preserving, so a permutation defeats it) and
    `_permute_map_rows_onto_window_table` cannot either (it requires every map
    row to be consumed, so extra rows defeat it). Before this existed the
    compound case fell through both and landed on the drop-record fallback,
    which removes rows by state_id and never looks at a centre -- so it kept the
    map's own contradicted row order and the result came back labelled a
    successful repair. A wrong mapping announced as REPAIRED is the worst
    outcome this module can produce.

    Then it turned out the IN-ORDER half of the same defect was still wide open,
    for the same reason one step along: the order-preserving matcher was
    answering that half by itself, greedily, so a phantom row duplicating a real
    window's centres and sitting earlier in the map got consumed for that window
    and the wrong mapping came back labelled REPAIRED with no reordering
    involved at all. Hence "every longer map" above. Whether the surviving rows
    happen to be in order is a question about WHICH repair happened, not about
    whether one is derivable, and the two must not be answered by the same
    call.

    FORCED, not greedy, and that is the whole difference from the sibling this
    one is otherwise a generalisation of. Two candidate rows for one window are
    resolved arbitrarily by a greedy first-unused-match, and the two cases are
    not equally survivable:

    * equal-count (`_permute_map_rows_onto_window_table`): both candidates are
      real windows OF THIS PHASE. They carry identical restraint centres, hence
      identical bias columns, so picking the wrong one mis-pools two sampled
      states and changes nothing else. Documented as that matcher's blind spot,
      measured at zero incidence on disk.
    * here: the map is longer than the window set, so a candidate may be a
      PHANTOM -- a row for a state this phase dropped and never sampled. Picking
      that one hands a real window's samples to a state that never ran, which is
      precisely the chignolin_6 failure (state 20, a phantom duplicate of 21,
      given 861,547 samples and its own f_k).

    So a window with two candidates, or two windows sharing one, is refused
    rather than resolved. Both refusals return a phrase naming the window and
    the state_ids involved, because the caller's whole job on that path is to
    say what it could not derive and why.

    Leftover map rows are the phase's phantoms and are returned to the caller's
    complement logic untouched -- deleting them re-creates the stale-registry
    bias bug for those states' MBAR columns (see `_PHANTOM_EPOCH_WINDOW_BASE`).
    """
    picked_idx = []
    claimed = {}
    for i, (c1, c2) in enumerate(real_centers):
        hits = [j for j, r in enumerate(map_rows)
                if _centers_close(_row_float(r, 'primary_center'), c1, tol)
                and _centers_close(_row_float(r, 'secondary_center'), c2, tol)]
        if not hits:
            return None, (f'local window {i} really ran at (primary {c1:.4f}, secondary {c2:.4f}) '
                          f'and no row of the map carries those centres in any position, so its '
                          f'state_id is recorded nowhere in these two artifacts and nothing can '
                          f'invent one')
        if len(hits) > 1:
            return None, (f'local window {i}\'s centres (primary {c1:.4f}, secondary {c2:.4f}) '
                          f'match {len(hits)} map rows (state_id '
                          f'{[_row_state_id(map_rows[j]) for j in hits]}), and with more rows than '
                          f'windows one of them may be a dropped state that never ran, so which '
                          f'state_id belongs to this window\'s samples is not derivable')
        j = hits[0]
        if j in claimed:
            return None, (f'local windows {claimed[j]} and {i} both ran at (primary {c1:.4f}, '
                          f'secondary {c2:.4f}) and the map holds a single row with those centres '
                          f'(state_id {_row_state_id(map_rows[j])}), so the two cannot be told '
                          f'apart and only one of them could be given a state_id')
        claimed[j] = i
        picked_idx.append(j)
    return [map_rows[j] for j in picked_idx], None


def _renumber_rows_in_order(rows: list, start: int = 0) -> list:
    """Copy of `rows` with ``epoch_window`` renumbered ``start..start+N-1`` in
    exactly the order given.

    Everything else on the row (state_id, primary/secondary centre, any k
    columns) is carried through untouched, and the inputs are never mutated.
    Split out from `_renumber_epoch_window_rows` because the permutation repair
    has already *chosen* the row order (the phase's own window table's) and
    re-sorting by the stale ``epoch_window`` values would put back precisely
    the order it just corrected.
    """
    out = []
    for i, r in enumerate(rows):
        new = dict(r)
        new['epoch_window'] = str(start + i)
        out.append(new)
    return out


def _renumber_epoch_window_rows(rows: list, start: int = 0) -> list:
    """Copy of `rows` with ``epoch_window`` renumbered ``start..start+N-1``, in
    ascending existing-``epoch_window`` order.

    With the default `start=0` this is exactly what the writer would have
    produced had it enumerated the phase's surviving windows instead of the
    registry's active states. `start` is used to park the *dropped* rows out of
    band -- see `_PHANTOM_EPOCH_WINDOW_BASE`.
    """
    return _renumber_rows_in_order(_sorted_map_rows(rows), start)


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


def _map_rows_disagreeing_with_window_table(map_rows: list, real_centers: list) -> list:
    """``[(local_window, (map_c1, map_c2), (real_c1, real_c2))]`` for every local
    window whose map row does NOT carry the restraint centres this phase really
    ran that window at. Empty when the two artifacts agree row for row.

    This is the membership check the row-count check cannot do. Both artifacts
    are written by the process that runs the phase, from the same post-drop
    in-memory window arrays (`gareus/production.py` rewrites
    ``umbrella_explicit_windows.csv`` immediately after the post-pull auto-drop
    and only then rewrites the map), so on a healthy phase they agree exactly,
    row for row -- measured on every adaptive-production phase holding samples
    under RUNS/: 117 of 117 phases whose row COUNT agrees also agree elementwise,
    0 disagree. Which is exactly why a disagreement is evidence rather than a
    heuristic: it is not a physical statistic with a tail, it is one of the
    phase's own two records of what it ran contradicting the other.

    Caller must pass equal-length inputs whose map rows are numbered 0..N-1
    (this compares local window *i* against map row *i* against table row *i*;
    the table's row order IS the post-drop local window order -- checked, not
    assumed, against the table's own ``window`` column by
    `_read_phase_window_table`). Both hold for
    every map any current writer produces -- the driver enumerates the registry,
    and every rewrite path renumbers the survivors -- and for all 152 maps under
    RUNS/; the caller checks rather than assumes, because comparing by position
    when the *consumer* resolves a sample by ``epoch_window`` value would let a
    non-contiguous map pass a check the samples then fail.

    What this cannot see, stated so it is not mistaken for a proof: two windows
    whose (primary, secondary) centres are *identical* can be exchanged without
    changing either artifact. Their restraint columns would be identical too, so
    the bias matrix is unharmed, but their samples would be pooled under each
    other's state_id. Nothing here claims to catch that -- and neither does any
    test on the sampled CVs, for the same reason.
    """
    bad = []
    for i, (c1, c2) in enumerate(real_centers):
        if i >= len(map_rows):
            break
        r = map_rows[i]
        m1 = _row_float(r, 'primary_center')
        m2 = _row_float(r, 'secondary_center')
        if not (_centers_close(m1, c1) and _centers_close(m2, c2)):
            bad.append((i, (m1, m2), (c1, c2)))
    return bad


def _map_cv2_corroboration(bad: list, cv2_means: dict) -> str:
    """One sentence of sampled-CV2 evidence for windows the centre check already
    flagged, or '' when there is no usable sampled cv2 for any of them.

    Deliberately reachable ONLY after `_map_rows_disagreeing_with_window_table`
    has found a disagreement, and deliberately not a firing rule of its own. An
    earlier revision of this check *was* a firing rule -- a per-window bar on
    |<cv2> - assigned centre| in units of that window's restraint width
    sqrt(kT/k2) -- and it does not survive contact with real data: sweeping the
    shipped implementation over every adaptive-production phase holding samples
    under RUNS/ (125 phases, 1,409 correctly-mapped windows) put 160 of those
    windows (11%) over its 3.0-sigma bar, firing on 47 phases that the
    deterministic centre comparison above finds nothing wrong with. Real
    umbrella windows sit systematically off their CV2 centre -- the pull of the
    underlying free energy, -(dG/dcv2)/k2, measured at a median 0.78 and a
    maximum 9.04 restraint widths on those runs -- while the CV2 centres
    themselves can be a small fraction of that apart (a 2D grid's windows differ
    mainly in CV1: chignolin_5's final/baseline has two windows at an identical
    secondary centre and five more within 0.02 of each other, against a typical
    0.1-0.7 offset). No statistic on the sampled cv2 mean, absolute or
    nearest-centre, can separate those two populations, and one that tries
    trains the operator to ignore the warning that matters.

    As corroboration it costs nothing and is worth quoting: the operator gets
    the two candidate centres and the number the samples actually produced,
    which is what tells them WHICH of the two artifacts to believe.
    """
    bits = []
    nearer_table = comparable = 0
    for ew, (_m1, m2), (_c1, c2) in bad:
        mu = cv2_means.get(ew)
        if mu is None or not math.isfinite(mu) or not (math.isfinite(m2) and math.isfinite(c2)):
            continue
        d_map, d_tab = abs(mu - m2), abs(mu - c2)
        comparable += 1
        if d_tab < d_map:
            nearer_table += 1
        if len(bits) < 5:
            bits.append(f'local window {ew} <cv2>={mu:.4f} (|cv2 - map centre| {d_map:.4f} vs '
                        f'|cv2 - table centre| {d_tab:.4f})')
    if not comparable:
        return ''
    return (f' Sampled CV2 for the affected windows: {"; ".join(bits)}; the samples are nearer the '
            f'window table\'s centre in {nearer_table} of {comparable} comparable window(s). '
            f'(Corroboration only -- a real window sits systematically off its own CV2 centre, up '
            f'to 9 restraint widths on healthy runs on disk, so the sampled mean cannot settle the '
            f'mapping by itself and is never what raised this note.)')


def _unchecked_membership_note(label: str, reason: str) -> WindowMapNote:
    """The "I could not run the membership check" note, in one place.

    Three different obstacles produce it (the two artifacts disagree about how
    many windows ran; either side's index column is not 0..N-1 in file order),
    and they must all say the same thing about their own weight: the row-count
    check passed, this one did not run, and that is NOT a detected fault.
    gareus_report.py grades this wording one band below a real membership
    failure, and `MAP_NOTE_UNCHECKED` says the same thing structurally to the
    consumers that must not treat it as one.
    """
    return WindowMapNote(
        f'[window map check] Could not cross-check the local-window -> state mapping of '
        f'adaptive-production phase {label} against its own surviving-window table: {reason}. '
        f'The row-count check (against this phase\'s recorded window count) passed; the '
        f'membership check did not run -- which is not the same as finding nothing wrong. '
        f'See {_STALE_WINDOW_MAP_DOC}.',
        MAP_NOTE_UNCHECKED)


# What the two membership notes below say about their own blind spots. Both
# limits are real and neither is closed by anything in this module:
#
#   * two windows whose (primary, secondary) centres are IDENTICAL are
#     interchangeable in both artifacts -- exchanging them changes neither, so
#     no comparison of centres can see it and the repair cannot tell which of
#     the two rows a given table row is. Their restraint columns are identical
#     so the bias matrix is unharmed, but their samples would be pooled under
#     each other's state_id. Zero incidence measured: no pair of windows shares
#     a full (primary, secondary) centre in any of the 291 window tables under
#     RUNS/.
#   * a row that keeps its centres and carries the wrong STATE_ID is invisible
#     here for the same reason -- the check compares centres, and both records
#     would still agree on those. The write-side check
#     (`repair_epoch_window_map_from_surviving_windows`, gareus/production.py)
#     compares the same two artifacts and is blind to it too.
#
# Stated in the note rather than only in this comment because the note is what
# an operator reads while deciding whether to trust a run.
_MEMBERSHIP_CHECK_SCOPE = (
    ' Verified here: each local window\'s (primary, secondary) restraint centre against the '
    'phase\'s own surviving-window table. NOT verified, and not closed by anything else either: '
    'two windows whose restraint centres are IDENTICAL are interchangeable in both records, so '
    'neither this check nor the repair can tell them apart; and a row that keeps its centres but '
    'carries the wrong state_id is invisible to a centre comparison, here and at write time.')


def _consistent_map_membership_notes(phase_dir, map_rows: list, cv2_means: dict) -> tuple:
    """``(rows, notes)`` for a map whose row COUNT agrees but whose *membership*
    may not. `rows` is None whenever nothing was rewritten.

    Returns `(None, [])` for the overwhelming majority of phases (the map lists
    exactly the windows the phase ran, in order). Otherwise one of three things
    happened, and the note says which:

    * the map lists the right windows in the WRONG ORDER -- a permutation. The
      correct mapping is derivable (see below), so the rows are rebuilt and a
      loud repair note comes back.
    * the map lists a genuinely DIFFERENT window set of the same size -- some
      window that really ran has no row at all. Nothing can re-derive its
      state_id, so the map is loaded unchanged with a loud warning.
    * a check could not run. Quieter note, and structurally not a fault.

    The evidence is the phase's own post-drop surviving-window table, compared
    row for row against the map (see
    `_map_rows_disagreeing_with_window_table`). That is the same comparison
    `gareus/production.py`'s `repair_epoch_window_map_from_surviving_windows`
    makes at write time, and that function's own comment names this file as
    where the analysis-side call belongs: the write-side check only ever runs in
    a process that (re)runs the phase, so it cannot protect an analysis of a run
    already on disk, which is every run this branch exists to rescue.

    Why a permutation is repairable and a different window set is not: local
    window *i*'s samples were generated at the centres the table records for
    row *i*, so the state_id that belongs to them is the one on the map row
    carrying those centres. When every table row has such a map row, that is a
    complete mapping and no inference is involved -- it is read off the same
    order-preserving centre match this check already performs to detect the
    problem, just without the order constraint. When a window that really ran
    has NO row in the map, its state_id is simply not recorded anywhere in
    these two artifacts, and no amount of matching invents it. (An earlier
    revision declined to repair either case and told the reader that a
    permutation was "a DIFFERENT window set" with "no row at all for a window
    that really ran" -- both false for a permutation, which is exactly the case
    this check was built to catch.)

    The repair is NOT gated on the sampled-CV2 fingerprint, deliberately. The
    fingerprint is quoted as evidence and never as a firing rule, for the
    reason `_map_cv2_corroboration` documents at length: real umbrella windows
    sit systematically off their own CV2 centre by up to 9 restraint widths,
    heterogeneously between neighbours, so `dev_after > dev_before` happens for
    a genuine permutation and a veto on it would decline a correct repair while
    asserting a physics claim that is not true. Which way it moves is not
    even a weak signal: on a fixture whose samples sit exactly on their own
    centres it collapses to zero under the repair, and on real data (where a
    window's mean CV2 is dragged off its centre by the underlying free energy)
    it can move the other way for the same correct reordering.

    Phases with no surviving-window table at all return `(None, [])` silently:
    such a phase never recorded what it ran (a plain 1D ladder, a legacy
    layout), so this is not a check that failed but a check that does not
    apply, and announcing a non-check once per such phase would be noise. Every
    pre-existing loader fixture in this repo has exactly that shape. A table
    that EXISTS but cannot be lined up by position is a different matter and
    does produce a note.
    """
    label = _phase_label(phase_dir)
    real_centers, table_problem = _read_phase_window_table(phase_dir)
    if real_centers is None:
        if table_problem is None:
            return None, []
        return None, [_unchecked_membership_note(label, table_problem)]
    if len(real_centers) != len(map_rows):
        # Unreachable on every real phase measured (125 of 125 have a window
        # table whose length equals this phase's authoritative window count, so
        # on this branch -- where that count already equals the map's row count
        # -- the table's length does too). Kept because if it ever does happen
        # the phase's own two records disagree about how many windows it ran,
        # which is not something to pass over in silence.
        return None, [_unchecked_membership_note(
            label, f'the table lists {len(real_centers)} window(s) while the map lists '
                   f'{len(map_rows)}, so the two cannot be compared row for row')]
    numbering = [_row_int(r, 'epoch_window') for r in map_rows]
    if numbering != list(range(len(map_rows))):
        # Not producible by any current writer (the driver enumerates, every
        # rewrite path renumbers 0..N-1) and not seen on any of the 152 maps
        # under RUNS/. Refuse to compare by position rather than quietly compare
        # the wrong pairs: a sample's local window index is resolved through the
        # `epoch_window` VALUE downstream, so position and value must agree
        # before a positional comparison means anything.
        return None, [_unchecked_membership_note(
            label, f'the map\'s epoch_window column is not numbered 0..{len(map_rows) - 1} '
                   f'({numbering[:8]}...), so its rows cannot be lined up with the table\'s '
                   f'by position')]
    bad = _map_rows_disagreeing_with_window_table(map_rows, real_centers)
    if not bad:
        return None, []
    worst = '; '.join(
        f'local window {ew}: map says (primary {m1:.4f}, secondary {m2:.4f}), the window table says '
        f'(primary {c1:.4f}, secondary {c2:.4f})'
        for ew, (m1, m2), (c1, c2) in bad[:5])
    more = f' (+{len(bad) - 5} more)' if len(bad) > 5 else ''

    reordered = _permute_map_rows_onto_window_table(map_rows, real_centers)
    if reordered is None:
        # Some window that really ran has no row in the map at all: the
        # signature of two attempts at this phase dropping equally many but
        # different windows (an interrupted phase re-pulled with a different
        # auto-drop verdict), which is the one residual the row-count check
        # cannot see. Warn rather than refuse: the artifacts do not say which
        # of the two is stale, refusing would make such a run permanently
        # unloadable, and a note is anything but quiet -- it carries into
        # meta['load_notes'], is printed by the union loader the moment it is
        # produced, lands in pmf_summary.json's warnings, is triaged HIGH by
        # gareus_report.py and FAILs that report's sample-to-state mapping
        # check.
        return None, [WindowMapNote(
            f'[window map check] epoch_window_map.csv for adaptive-production phase {label} has '
            f'the right NUMBER of rows ({len(map_rows)}) for the windows this phase ran, but '
            f'{len(bad)} of them carry a different restraint centre than the phase\'s own '
            f'post-drop window table (umbrella_explicit_windows.csv) records for that local '
            f'window: {worst}{more}. At least one window that really ran has NO row in the map, '
            f'so this is not a reordering: the map describes a DIFFERENT window set of the same '
            f'size. The map is loaded unchanged, because a window with no row has no recorded '
            f'state_id anywhere in these two artifacts and nothing can re-derive one. This '
            f'phase\'s samples may therefore be attributed to the wrong umbrella state -- treat '
            f'every free energy derived from them as invalid until the mapping is confirmed by '
            f'hand.{_map_cv2_corroboration(bad, cv2_means)}{_MEMBERSHIP_CHECK_SCOPE} '
            f'See {_STALE_WINDOW_MAP_DOC}.',
            MAP_NOTE_FAULT_UNREPAIRED)]

    repaired = _renumber_rows_in_order(reordered)
    shifted = [f'{i}->{_row_state_id(r)}' for i, r in enumerate(repaired)
               if _row_state_id(r) != _row_state_id(map_rows[i])]
    dev_before = _map_cv2_fingerprint_deviation(map_rows, cv2_means)
    dev_after = _map_cv2_fingerprint_deviation(repaired, cv2_means)
    if dev_before is not None and dev_after is not None:
        evidence = (f' Sampled-cv2 fingerprint across the reordering: mean |cv2 - '
                    f'secondary_center| {dev_before:.4f} -> {dev_after:.4f} (evidence, not the '
                    f'reason: real windows sit systematically off their own CV2 centre -- up to 9 '
                    f'restraint widths on healthy runs on disk, and by different amounts between '
                    f'neighbours -- so this number can move either way for a genuine reordering '
                    f'and never decides one).')
    else:
        evidence = (' No sampled cv2 was available to cross-check the reordering against '
                    '(CV1-only phase, or no secondary centres in the map).')
    return repaired, [WindowMapNote(
        f'[stale window map] epoch_window_map.csv for adaptive-production phase {label} has the '
        f'right NUMBER of rows ({len(map_rows)}) for the windows this phase ran and lists exactly '
        f'the windows its own post-drop window table (umbrella_explicit_windows.csv) records, but '
        f'{len(bad)} of them sit at the wrong local window: {worst}{more}. The map is therefore a '
        f'PERMUTATION of the right rows -- every window that really ran does have a row, only the '
        f'order is wrong -- so the correct mapping IS derivable and was REPAIRED IN MEMORY by '
        f'matching each map row onto the table row carrying the same restraint centres. '
        f'Corrected local->state mapping for the {len(shifted)} moved window(s): '
        f'{", ".join(shifted)}. Every sample in those windows was previously attributed to the '
        f'wrong umbrella state, so any earlier analysis of this phase is invalid.{evidence} The '
        f'run data on disk is unchanged and still stale.{_MEMBERSHIP_CHECK_SCOPE} '
        f'See {_STALE_WINDOW_MAP_DOC}.',
        MAP_NOTE_REPAIRED)]


def _validate_and_repair_epoch_window_map(phase_dir, wmap_rows: list,
                                          window_ids=None, cv2=None) -> tuple:
    """``(rows, notes)`` for one adaptive-production phase's window map.

    Returns `wmap_rows` unchanged when the map's row count agrees with how many
    windows the phase really ran -- the overwhelming majority of phases,
    including every phase of every run that never used
    `--us-auto-drop-bad-windows`. That path is still not free of notes, and can
    itself repair: an agreeing row count does not prove agreeing *membership*,
    so the map's own restraint centres are compared row for row against the
    phase's post-drop surviving-window table, which yields a repair when the
    map lists the right windows in the wrong order, a warning when it lists a
    genuinely different window set of the same size, and a quieter note when
    that comparison could not be made at all (see
    `_consistent_map_membership_notes`). Clean phases return no notes.

    Every note carries a `kind` (see `WindowMapNote` and the MAP_NOTE_*
    constants): a non-empty `notes` is NOT a boolean for "the map was stale",
    and a consumer that treats it as one refuses runs whose map is not known to
    be wrong at all. Branch on `window_map_note_reports_a_fault` /
    `window_map_note_rewrote_rows`, on the note object itself, before any
    f-string copy of its text.

    On a mismatch the map is stale (see this section's header comment) and the
    phantom rows are identified from whichever of two independent records the
    phase left behind -- preferring the direct one:

    1. the phase's own surviving window table (``umbrella_explicit_windows.csv``),
       matched onto the map by restraint centre -- this is what actually ran, and
       it is the only source that covers a phase whose drop record is missing or
       does not account for the whole gap (both occur on real runs on disk).
       Accepted only when every real window has exactly one map row carrying its
       centres, so the mapping is READ OFF rather than inferred -- in the map's
       own row order, or (for a map that also lists its windows out of order) in
       the table's, one gate for both, because with more rows than windows an
       ambiguous candidate can be a state the phase dropped and never sampled
       (see `_select_map_rows_onto_window_table`). If that does not hold, this
       fails closed WITHOUT trying source 2: a record that identifies rows by
       position in this same map can neither restore an order that map has
       already contradicted nor break a tie between two of its own rows;
    2. failing that -- i.e. only when there is no usable table to contradict
       anything -- the post-pull ``dropped_post_pull_bad_windows`` record -- the
       phase's own, or an inherited sibling sub-run's -- applied by state_id and
       accepted only if the surviving row count reconciles.

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
        # The count agrees, but agreeing on the count is not the same as
        # agreeing on the membership. Cross-check the map's own restraint
        # centres, row for row, against the phase's post-drop surviving-window
        # table, which is the other record of what this phase really ran (see
        # _consistent_map_membership_notes: it repairs a pure reordering, warns
        # without repairing when a window that ran has no row at all, and says
        # so quietly when a check could not run).
        #
        # `list(wmap_rows)` -- the caller's own rows, in their own order -- is
        # returned untouched whenever nothing was rewritten, exactly as before:
        # the healthy path must stay byte-identical, and `rows` above is a
        # re-sorted copy made only so the checks can rely on an order.
        membership_rows, membership_notes = _consistent_map_membership_notes(
            phase_dir, rows, cv2_means)
        if membership_rows is None:
            return list(wmap_rows), membership_notes
        return membership_rows, membership_notes

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
            return list(wmap_rows), [WindowMapNote(
                f'[stale window map] {detail}. {reason} Loaded with the STALE map anyway because '
                f'{_STALE_WINDOW_MAP_OVERRIDE_ENV} is set: this phase\'s samples are attributed to '
                f'the wrong umbrella states and every free energy derived from them is invalid. '
                f'See {_STALE_WINDOW_MAP_DOC}.',
                MAP_NOTE_STALE_LOADED)]
        raise ValueError(msg)

    # Preferred repair source: the phase's own surviving window table, matched
    # onto the map by centre (direct evidence of what ran -- see
    # _phase_real_window_centers). Only trusted when its length agrees with the
    # phase's real window count, otherwise it is itself stale/partial.
    #
    # `kept_rows` is the rows of `rows` belonging to windows that really ran, in
    # the order the repair will use them (a subsequence of `rows` unless the map
    # also lists its windows out of order); `phantom_src_rows` is everything else. Both halves are needed: only
    # the first may appear in the local-index lookup, and both must survive
    # into the returned list so the state-keyed native-params parser still sees
    # the dropped states' epoch-native window params (see
    # _PHANTOM_EPOCH_WINDOW_BASE -- deleting them re-creates the stale-registry
    # bias bug for those states' columns).
    kept_rows = None
    repair_src = ''
    real_centers = _phase_real_window_centers(phase_dir)
    if real_centers is not None and len(real_centers) == n_true:
        # ONE gate decides whether these two artifacts FORCE a local-window ->
        # state_id assignment at all, and it is asked BEFORE the map's row order
        # is looked at: `_select_map_rows_onto_window_table`, which requires each
        # real window to have exactly one map row carrying its restraint centres
        # and refuses anything less. The row order is consulted afterwards, and
        # only to say WHICH repair happened -- never to decide whether one is
        # derivable.
        #
        # That order used to be the other way round: the order-preserving greedy
        # matcher answered first and the forced one was reached only as its
        # fallback (for a map that is both longer than the window set AND lists
        # it out of order). Which left greedy deciding the ambiguous IN-ORDER
        # case alone, and greedy is precisely what must not decide it:
        # `_match_map_rows_to_real_windows` takes the first row whose centres
        # match, so a PHANTOM row -- one for a state this phase dropped and never
        # sampled -- carrying a real window's centres and sitting earlier in the
        # map is consumed for that window, the remaining windows still match, and
        # the wrong mapping comes back labelled REPAIRED. Three real windows,
        # four map rows, the phantom first: measured on a fixture, not argued (see
        # tests/test_equal_count_map_fingerprint.py::test_the_loader_refuses_an_
        # in_order_longer_map_whose_first_row_is_an_ambiguous_phantom). A wrong
        # mapping announced as REPAIRED is the worst outcome this module can
        # produce, and it was reachable on both orderings while only one of them
        # was guarded.
        #
        # So the refusal is SHARED rather than duplicated: two subtly different
        # answers to "is this ambiguous" on the two orderings of one defect is
        # the drift these paired checks exist to remove. The equal-count path is
        # deliberately NOT routed through here and is not made stricter: with as
        # many rows as windows every candidate is a real window OF THIS PHASE
        # carrying identical restraint columns, so an arbitrary pick mis-pools two
        # sampled states and changes nothing else, whereas here it can hand a real
        # window's samples to a state that never ran. That asymmetry is stated at
        # length in `_select_map_rows_onto_window_table`'s own two bullets.
        #
        # The write side reaches the same verdict from the same two files
        # (`repair_epoch_window_map_from_surviving_windows`,
        # gareus/production.py), which is checked by running BOTH on
        # byte-identical copies of one phase directory and comparing the mappings
        # element for element, rather than argued about -- see
        # tests/test_equal_count_map_fingerprint.py::
        # test_both_sides_derive_the_same_mapping_for_a_compound_map and its two
        # refusal twins. A previous round asserted that agreement by reading the
        # two implementations and was wrong.
        #
        # Free on real data, measured rather than assumed: over all 152 maps under
        # RUNS/, each of the 8 phases that needs this repair has its selection
        # FORCED, and the rows the forced matcher picks are the identical row
        # objects the greedy one picked -- 8 repaired / 0 refused, before and
        # after, with byte-identical notes.
        selected, problem = _select_map_rows_onto_window_table(rows, real_centers)
        # Diagnosis, never the verdict: are the rows the real windows match sitting
        # in the map's own row order? A plain drop -- the shape of every affected
        # phase on disk -- says yes, the compound defect says no. Asked of the map
        # as it is, so it is answerable on the refusal path too, where all it
        # decides is which of the two refusals below describes its own evidence
        # truthfully.
        in_order = _match_map_rows_to_real_windows(rows, real_centers) is not None
        if selected is None:
            # Two refusals, because they have different evidence and a refusal
            # that misdescribes its own evidence is this same failure shape one
            # step removed: out of order, the window table has contradicted the
            # map's row order; in order, the order agrees and it is the tie that
            # cannot be broken.
            #
            # The drop record is not consulted as a fallback from either. Out of
            # order the reason is that it says nothing about centres, so applying
            # it would preserve exactly the row order the table has just
            # contradicted and hand that back under a REPAIRED note. In order it
            # would in principle name which of two duplicate-centre rows is the
            # phantom -- but it identifies rows by POSITION IN THIS VERY MAP (and
            # may be inherited from a sibling sub-run; see
            # `_dropped_state_ids_for_phase`) and is only ever count-checked, so
            # leaning on it to break a centre tie is deciding by fiat, i.e. the
            # repair-on-a-guess this guard exists not to publish. The write side
            # abstains on the same input for the same reason
            # (`_has_duplicate_center_pairs`, gareus/production.py, which refuses
            # before its own in-order matcher runs), so refusing here keeps the
            # two sides agreeing on this input class instead of opening a new gap.
            if in_order:
                return _fail(
                    f"Its own surviving window table (umbrella_explicit_windows.csv) records the "
                    f"{n_true} window(s) it really ran, and the map's rows do carry those windows' "
                    f"restraint centres in order -- but that reading does not settle WHICH rows the "
                    f"extra ones are: {problem}. With more rows than windows an extra row can be a "
                    f"state this phase dropped and never sampled, so taking the first row whose "
                    f"centres match would risk attributing a real window's samples to a state that "
                    f"never ran. The drop record is deliberately not used as a fallback either -- it "
                    f"identifies rows by position in this same map and cannot say which of two rows "
                    f"carrying one window's centres is the phantom.")
            return _fail(
                f"Its own surviving window table (umbrella_explicit_windows.csv) records the "
                f"{n_true} window(s) it really ran, and the map's rows do not carry those "
                f"windows' restraint centres in order, so the extra rows cannot simply be "
                f"removed. Neither can the mapping be read off out of order: {problem}. The "
                f"drop record is deliberately not used as a fallback once the window table "
                f"has contradicted the map's row order -- it says nothing about centres, so "
                f"it would preserve exactly the order in question.")
        kept_rows = selected
        # The in-order wording is unchanged, deliberately: it is what every
        # affected phase on disk produces, and its note must stay byte-identical
        # across this change.
        repair_src = ('its own surviving window table (umbrella_explicit_windows.csv)'
                      if in_order else
                      'its own surviving window table (umbrella_explicit_windows.csv), matched '
                      'onto the map OUT OF ORDER: the map both lists more windows than ran and '
                      'lists them in the wrong order, and every real window had exactly one '
                      'map row carrying its restraint centres')

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
    # `_renumber_rows_in_order`, NOT `_renumber_epoch_window_rows`: the order
    # `kept_rows` is already in is the answer, and re-sorting by the stale
    # `epoch_window` values would put back precisely the order the compound
    # matcher just corrected. A provable no-op wherever the chosen rows ARE a
    # subsequence of `rows` -- which `_sorted_map_rows` has already put in
    # ascending `epoch_window` order, so re-sorting a subsequence of it is the
    # identity: the drop-record fallback by construction (a filter over `rows`),
    # and the window-table selection exactly in the `in_order` case above, which
    # is the shape of every affected phase on disk. The phantoms keep the sorting
    # version: their order is unobservable (they are
    # parked out of band where no sample can reach them) and their own
    # `epoch_window` values are the only order they have.
    survivors = _renumber_rows_in_order(kept_rows)
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
    #
    # Note this veto is a firing rule on THIS path, unlike the equal-count
    # membership path where the same number is quoted as evidence only (a real
    # window sits systematically off its own CV2 centre, by different amounts
    # between neighbours, so the deviation can rise for a genuine reordering).
    # The consequence is deliberate and one-directional: a correct compound
    # repair whose fingerprint happens to worsen is REFUSED rather than
    # published. That costs an operator one `GAREUS_ALLOW_STALE_WINDOW_MAP=1`
    # inspection run; the opposite error costs a wrong free energy.
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
    note = WindowMapNote(
        f'[stale window map] {detail} -- `--us-auto-drop-bad-windows` dropped windows post-pull '
        f'and renumbered the survivors 0..{n_true - 1}, but this map was never rewritten. '
        f'REPAIRED IN MEMORY from {repair_src}; {impact}'
        f'{evidence}{phantom_note} The run data on disk is unchanged and still stale; see '
        f'{_STALE_WINDOW_MAP_DOC}.',
        MAP_NOTE_REPAIRED)
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
    v_pep_list, v_dih_list, lam_list = [], [], []
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
        try: v_pep_list.append(float(row.get('v_pep_kj_mol', '') or 'nan'))
        except Exception: v_pep_list.append(float('nan'))
        try: v_dih_list.append(float(row.get('v_dih_kj_mol', '') or 'nan'))
        except Exception: v_dih_list.append(float('nan'))
        try: lam_list.append(float(row.get('gamd_lambda', '') or 0.0))
        except Exception: lam_list.append(float('nan'))

    cv = np.asarray(cv_list, dtype=np.float64)
    cv2 = np.asarray(cv2_list, dtype=np.float64)
    rg = np.asarray(rg_list, dtype=np.float64)
    window = np.asarray(win_list, dtype=int)
    replica = np.asarray(rep_list, dtype=int)
    step = np.asarray(step_list, dtype=int)
    boost = np.asarray(boost_list, dtype=np.float64)
    pot = np.asarray(pot_list, dtype=np.float64)
    v_pep = np.asarray(v_pep_list, dtype=np.float64)
    v_dih = np.asarray(v_dih_list, dtype=np.float64)
    lam_sample = np.asarray(lam_list, dtype=np.float64)
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

    # lambda-ladder: per-merged-state rung, derived from the per-sample
    # gamd_lambda column by grouping on this loader's own merged window
    # index (no registry/window-snapshot here carries a per-state
    # gamd_lambda -- same derivation as gareus.mbar_analysis.loaders'
    # load_csv/load_parquet). A no-op through apply_ladder_boost_to_u when
    # no state carries gamd_lambda > 0.
    state_lambdas = np.zeros(K, dtype=np.float64)
    if np.any(np.isfinite(lam_sample)):
        for k in range(K):
            grp = lam_sample[(window == k) & np.isfinite(lam_sample)]
            if grp.size:
                state_lambdas[k] = float(np.nanmedian(grp))
    from .ladder import apply_ladder_boost_to_u, load_pep_gamd_envelope
    _envelope = load_pep_gamd_envelope(ap) if np.any(state_lambdas > 0.0) else None
    u_nk = apply_ladder_boost_to_u(u_nk, v_pep, v_dih, state_lambdas, _envelope, beta, meta)

    meta['load_notes'] = [f'Loaded {cv.size} samples from {len(sources)} epoch CSV sources; union {K} windows.']
    meta['umbrella_window_rows'] = [
        {'center_A': str(centers[i]), 'k_kcal_mol_A2': str(k_kcal[i])} for i in range(K)
    ]
    meta['adaptive_epoch_run_dirs'] = [str(s) for s in sources]
    meta['_epoch_source'] = src_idx_list
    src_str = f'{sources[0]}/samples.csv ... {sources[-1]}/samples.csv'
    return clean(Data(root, root / 'pmf_analysis', cv, cv2, rg, window, replica, step, u_nk, centers, k_kcal, beta, temp, boost, pot, src_str, meta,
                       v_pep_kj=v_pep, v_dih_kj=v_dih, state_lambdas=state_lambdas))


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
